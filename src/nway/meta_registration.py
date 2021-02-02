import numpy as np
from skimage.metrics import structural_similarity
from functools import partial
from typing import List, Union, Callable, Tuple
import nway.image_processing_utils as impu
import nway.transforms as tf


class MetaRegistration():
    """Attempts multiple types of registration and chooses the best
    one based on an evaluation metric.

    Attributes
    ----------
    candidates: List[tf.TransformList]
        the candidate registration pipelines. Each one is a TransformList
    scores: List[float]
        the evaluation scores associated with each candidate
    transformed_images: List[List[np.ndarray]]
        each entry is the resulting transformed images from the candidate
        pipelines. NOTE: both src and dst can be modified by a transform.
    best_score: float
        the best score of all candidates
    best_matrix: np.ndarray
        the accumulated transform matrix for the best pipeline
    best_candidate: List[str]
        list of names of the transforms from the best candidate

    Methods
    -------
    estimate(src, dst):
        estimates the transforms for all the candidates
    evaluates(src, dst):
        applies the transforms for all the candidates and scores
    select():
        evaluates the scoring of the candidates and sets best attributes
    measure(src, dst):
        perform metric evaluation on a pair of images

    Notes
    -----
    This meta-registration approach accounts for a diverse set of input
    images that were difficult to align with a single strategy. If a new set
    of images is not successfully registered by one of these candidates,
    additional candidate lists can be added, as well as additional Transforms

    """
    def __init__(self, maxCount: int = 1000, epsilon: float = 1.5e-7,
                 motion_type: str = 'MOTION_EUCLIDEAN',
                 gaussFiltSize: int = 5, CLAHE_grid: int = 24,
                 CLAHE_clip: float = 2.5, edge_buffer: int = 40,
                 ssim_sigma: float = 20.0, include_original: bool = False):
        """

        Parameters
        ----------
        maxCount: int
            passed as termination criterion for cv2.findTransformECC
        epsilon: float
            passed as termination criterion for cv2.findTransformECC
        motion_type: str
            used to specify motion_type (only cv2.findTransformECC
            contributes non-Euclidean matrix elements)
        gaussFiltSize: int
            must be odd. passed to cv2.findTransformECC
        CLAHE_grid: int
            passed as tileGridSize to cv2.createCLAHE.
        CLAHE_clip : float
            passed as clipLimit to cv2.createCLAHE
        edge_buffer: int
            this many pixels will be cropped from top, bottom, left, and right
            of both images in the crop() steps.
        ssim_sigma: float
            passed to skimage.metrics.structural_similarity as sigma, with
            gaussian_weights=True. Larger number dampens high frequency
            content of this measure and is thus less sensitive to
            real biological variations.
        include_original: bool
            whether to include the original strategy of CLAHE + ECC which
            occasionally gave a slight registration improvement, but not
            very good. Including for legacy (and legacy regression test)

        """

        # make these partials, so we can succinctly instantiate new ones
        contrast = partial(tf.CLAHE, CLAHE_grid=CLAHE_grid,
                           CLAHE_clip=CLAHE_clip)
        phase = partial(tf.PhaseCorrelate)
        crop = partial(tf.Crop, edge_buffer=edge_buffer)
        ecc = partial(tf.ECC, maxCount=maxCount, epsilon=epsilon,
                      motion_type=motion_type, gaussFiltSize=gaussFiltSize)

        # instantiate one contrast() as attribute for using in evaluate()
        self.contrast = contrast()

        self.motion_type = motion_type

        # function to evaluate alignment
        self.measure = partial(structural_similarity,
                               gaussian_weights=True,
                               sigma=ssim_sigma)

        # in the event that no transform is the best alignment (unlikely)
        # this just makes the reported name `Identity` rather than `Transform`
        class Identity(tf.Transform):
            pass

        # independently attempt alignment multiple ways
        self.candidates = [
                # Identity is unlikely as the best candidate
                # and probably indicates some data problem.
                tf.TransformList(transforms=[Identity()]),
                # cropping eliminates some motion-correction induced
                # borders that can bias the registration
                tf.TransformList(transforms=[crop(), ecc()]),
                # sometimes, ECC works better after contrast adjustment
                tf.TransformList(transforms=[crop(), contrast(), ecc()]),
                # sometimes, just PhaseCorrelation is a good estimator
                tf.TransformList(transforms=[crop(), contrast(), phase()]),
                # sometimes, using the PhaseCorrelation translations
                # as a precursor to ECC is helpful
                tf.TransformList(transforms=[crop(), contrast(), phase(),
                                             ecc()])
                ]
        if include_original:
            self.candidates.append(
                    tf.TransformList(transforms=[contrast(), ecc()]))

    def __call__(self, src, dst):
        """attempts all registration candidates, evaluates them,
        and selects the best based on measure()

        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        """
        self.candidates, self.failed = self.estimate(self.candidates, src, dst)
        self.scores = self.evaluate(self.candidates, self.failed,
                                    self.motion_type, self.contrast,
                                    self.measure, src, dst)
        index = self.select(self.scores)
        self.best_matrix = self.candidates[index].matrix
        self.best_candidate = [tf.__class__.__name__
                               for tf in self.candidates[index].transforms]

    @staticmethod
    def estimate(candidates, src, dst
                 ) -> Tuple[List[tf.TransformList], List[bool]]:
        """estimates the transforms in each candidate TransformList
        in the event of an exception, notes that candidate as failed.

        Parameters
        ----------
        candidates: List[TransformList]
            candidate TransformLists to evaluate
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        Returns
        -------
        candidates: List[TransformList]
            candidate TransformLists, which have been evaluated
        failed: List[bool]
            same length/order as candidates. Whether the evaluation failed

        """
        failed = [False] * len(candidates)
        for ic, candidate in enumerate(candidates):
            try:
                candidate.estimate(src, dst)
            except tf.TransformException:
                failed[ic] = True
        return candidates, failed

    @staticmethod
    def evaluate(candidates: List[tf.TransformList], failed: List[bool],
                 motion_type: str, contrast: tf.CLAHE,
                 measure: Callable[[np.ndarray, np.ndarray], float],
                 src: np.ndarray, dst: np.ndarray) -> List[Union[float, None]]:
        """evaluates the candidate TransformLists by applying their
        estimated transforms, warping the src image and measuring the
        ssim parameter.

        Parameters
        ----------
        candidates: List[TransformList]
            candidate TransformLists to evaluate
        failed: List[bool]
            whether a particular candidate failed to estimate
        motion_type: str
            motion_type for warping moving image
        contrast: tf.CLAHE
            an instance of CLAHE to apply before warping and evaluating
        measure: function
            evaluates the metric of two images
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        Returns
        -------
        scores: List[float]
             the evaluation scores. Can be None for a candidate that
             raised an exception.

        Note
        ----
        evaluations can only be compared across candidates
        by comparing ssim after geometrical transforming only.
        Attempting to compare ssim between the entire transforms
        will likely not work well, as things like contrast adjustment
        and cropping will change the measure differentially.

        """
        scores = []
        for candidate, ifail in zip(candidates, failed):
            if ifail:
                scores.append(None)
            else:
                mov, fix = contrast.apply(src, dst)
                mov = impu.warp_image(mov, candidate.matrix,
                                      motion_type, fix.shape)
                scores.append(measure(mov, fix))
        return scores

    @staticmethod
    def select(scores: List[Union[float, None]]) -> int:
        """select the best candidate based on the maximum
        ssim metric.

        Parameters
        ----------
        scores: List[float]
            scores from evaluating the candidates. Can contain None

        Returns
        -------
        index: int
            index of best score

        """
        index = np.nanargmax(np.array(scores, dtype='float'))
        return index
