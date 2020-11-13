import cv2
import numpy as np
import logging
from typing import List, Tuple
import nway.image_processing_utils as impu

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransformException(Exception):
    pass


class Transform():
    """identity transform and base class for other transforms.

    Attributes
    ----------
    matrix: np.ndarray
        3 x 3 transformation matrix

    Methods
    -------
    estimate(src, dst):
        estimates the transform matrix. For generality, all transforms
        accept src and dst for estimate method, even when unused, as here
    apply(src, dst):
        applies the transform to src (moving image). For generality, both
        src and dst are potentially modified by apply()

    """
    def __init__(self):
        return

    def estimate(self, src: np.ndarray, dst: np.ndarray):
        """estimates transform matrix given src and dst.
        In this case, just identity.

        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        """
        self.matrix = np.eye(3).astype('float32')

    def apply(self, src: np.ndarray, dst: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """applies a Euclidean transform to src with self.matrix

        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        Returns
        -------
        mov: np.ndarray
            uint8, transformed src image
        dst: np.ndarray
            uint8, transformed dst image. in this case, not transformed

        """
        mov = impu.warp_image(src, self.matrix, 'MOTION_EUCLIDEAN', dst.shape)
        return mov, dst


class CLAHE(Transform):
    """applies contrast limited adaptive histogram equalization
    to both src and dst images. Has no geometrical transform, yet
    inherits from Transform() to be easily swappable in a transform List.
    """
    def __init__(self, CLAHE_grid: int, CLAHE_clip: float):
        """
        Parameters
        ----------
        CLAHE_grid : int
            passed as tileGridSize to cv2.createCLAHE
        CLAHE_clip : float
            passed as clipLimit to cv2.createCLAHE
        """
        self.CLAHE_grid = CLAHE_grid
        self.CLAHE_clip = CLAHE_clip

    def apply(self, src: np.ndarray, dst: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """applies CLAHE to both src and dst

        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        Returns
        -------
        mov: np.ndarray
            uint8, contrast-adjusted src image
        dst: np.ndarray
            uint8, contrast-adjusted dst image.

        """
        mov = impu.contrast_adjust(src, self.CLAHE_grid, self.CLAHE_clip)
        fix = impu.contrast_adjust(dst, self.CLAHE_grid, self.CLAHE_clip)
        return mov, fix


class PhaseCorrelate(Transform):
    """estimates geometrical transform by phase correlation
    and applies the transform
    """
    def estimate(self, src: np.ndarray, dst: np.ndarray):
        """
        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        """
        (dx, dy), _ = cv2.phaseCorrelate(src1=dst.astype('float32'),
                                         src2=src.astype('float32'))
        self.matrix = np.eye(3).astype('float32')
        self.matrix[0, 2] = dx
        self.matrix[1, 2] = dy

    def apply(self, src: np.ndarray, dst: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        mov = impu.warp_image(src, self.matrix, 'MOTION_EUCLIDEAN', dst.shape)
        return mov, dst


class Crop(Transform):
    """crops both src and dst image, default identity transform
    """
    def __init__(self, edge_buffer: int):
        """

        Parameters
        ----------
        edge_buffer: int
            this many pixels will be cropped from top, bottom, left, and right

        """
        self.edge_buffer = edge_buffer

    def apply(self, src: np.ndarray, dst: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        Returns
        -------
        cropped_src: np.ndarray
            uint8, cropped src image
        cropped_dst: np.ndarray
            uint8, cropped dst image.

        """
        if self.edge_buffer == 0:
            return src, dst
        if np.any(self.edge_buffer >
                  0.5 * np.array((src.shape, dst.shape)).flatten()):
            raise TransformException(f"edge buffer of {self.edge_buffer} "
                                     "is too large for images of shapes "
                                     f"{src.shape}, {dst.shape}")
        cropped_src = np.copy(src[
            self.edge_buffer:-self.edge_buffer,
            self.edge_buffer:-self.edge_buffer])
        cropped_dst = np.copy(dst[
            self.edge_buffer:-self.edge_buffer,
            self.edge_buffer:-self.edge_buffer])
        return cropped_src, cropped_dst


class ECC(Transform):
    """estimates a transform between two images with cv2.findTransformECC
    and applies the transform to one of the images.
    """
    def __init__(self, maxCount: int, epsilon: float,
                 motion_type: str, gaussFiltSize: int):
        self.maxCount = maxCount
        self.epsilon = epsilon
        self.gaussFiltSize = gaussFiltSize
        self.motion_type = motion_type

    @classmethod
    def cv2_motion(self, motion_type: str) -> int:
        cvmotion = {
                "MOTION_TRANSLATION": cv2.MOTION_TRANSLATION,
                "MOTION_EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
                "MOTION_AFFINE": cv2.MOTION_AFFINE,
                "MOTION_HOMOGRAPHY": cv2.MOTION_HOMOGRAPHY}
        return cvmotion[motion_type]

    def estimate(self, src: np.ndarray, dst: np.ndarray):
        """estimates a transform from src to dst with
        cv2.findTransformECC
        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        """
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.maxCount, self.epsilon)
        tform = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]]).astype('float32')
        if self.motion_type == 'MOTION_HOMOGRAPHY':
            hrow = np.array([0.0, 0.0, 1.0]).astype('float32')
            tform = np.vstack((tform, hrow))
        try:
            # Run the ECC algorithm. The results are stored in tform
            ccval, tform = cv2.findTransformECC(
                    templateImage=dst,
                    inputImage=src,
                    warpMatrix=tform,
                    motionType=self.cv2_motion(self.motion_type),
                    criteria=criteria,
                    inputMask=None,
                    gaussFiltSize=self.gaussFiltSize)
        except cv2.error as cv2err:
            err = TransformException("findTransformECC failed to align images")
            err.args = err.args + cv2err.args
            raise err

        if tform.shape == (2, 3):
            tform = np.vstack((tform, [0, 0, 1]))
        self.matrix = tform

    def apply(self, src: np.ndarray, dst: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        Returns
        -------
        mov: np.ndarray
            uint8, transformed src image
        dst: np.ndarray
            original dst image

        """
        mov = impu.warp_image(src, self.matrix, self.motion_type, dst.shape)
        return mov, dst


class TransformList():
    """a list of Transforms that behaves like a Transform
    """
    def __init__(self, transforms: List[Transform]):
        """

        Parameters
        ----------
        transforms: List[Transform]
            List of Transform instances, or Transform-based instances

        """
        self.transforms = transforms

    def estimate(self, src, dst):
        """estimates the individual transforms by successively
        estimating and applying them.

        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        """
        mov = np.copy(src)
        fix = np.copy(dst)
        for transform in self.transforms:
            transform.estimate(mov, fix)
            mov, fix = transform.apply(mov, fix)

    def apply(self, src, dst
              ) -> Tuple[np.ndarray, np.ndarray]:
        """successively applies transforms to a pair of images

        Parameters
        ----------
        src: np.ndarray
            uint8, needed for opencv operations
        dst: np.ndarray
            uint8, needed for opencv operations

        Returns
        -------
        mov: np.ndarray
            uint8, transformed src image
        fix: np.ndarray
            uint8, transformed dst image

        """
        mov = np.copy(src)
        fix = np.copy(dst)
        for transform in self.transforms:
            mov, fix = transform.apply(mov, fix)
        return mov, fix

    @property
    def matrix(self) -> np.ndarray:
        """the cumulative geometric transform matrix
        for the list of transforms

        Returns
        -------
        tform: np.ndarray
            3x3 float32 transform matrix
        """
        tform = np.eye(3).astype('float32')
        for transform in self.transforms:
            tform = transform.matrix.dot(tform)
        return tform
