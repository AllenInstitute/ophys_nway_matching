import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def register_image_pair(img_fixed: np.ndarray, img_moving: np.ndarray,
                        maxCount: int, epsilon: float,
                        motion_type: str, gaussFiltSize: int,
                        preregister: bool = True) -> np.ndarray:
    """find the transform that registers two images

    Parameters
    ----------
    img_fixed : numpy.ndarray
        CV_8U (uint8) or CV_32F (float32) fixed image
    img_moving : numpy.ndarray
        CV_8U (uint8) or CV_32F (float32) moving image
    maxCount : int
        passed as maxCount to opencv termination criteria
    epsilon : float
        passed as epsilon to opencv termination criteria
    motion_type : str
        one of the 4 possible motion types for opencv findTransformECC
        see the dictionary cvmotion below for the 4 possible values
    gaussFiltSize : int
        passed to opencv findTransformECC(). An optional value
        indicating size of gaussian blur filter
    preregister: bool
        whether to give a hint to cv2.findTransformECC from cv2.phaseCorrelate

    Returns
    -------
    tform : :class:`numpy.ndarry`
        3 x 3 transformation matrix

    """

    cvmotion = {
            "MOTION_TRANSLATION": cv2.MOTION_TRANSLATION,
            "MOTION_EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
            "MOTION_AFFINE": cv2.MOTION_AFFINE,
            "MOTION_HOMOGRAPHY": cv2.MOTION_HOMOGRAPHY}

    # Define termination criteria
    criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            maxCount,
            epsilon)

    dx = dy = 0.0
    if preregister:
        (dx, dy), _ = cv2.phaseCorrelate(src1=img_fixed.astype('float32'),
                                         src2=img_moving.astype('float32'))

    tform = np.array([[1.0, 0.0, dx],
                      [0.0, 1.0, dy]]).astype('float32')
    if motion_type == 'MOTION_HOMOGRAPHY':
        hrow = np.array([0.0, 0.0, 1.0]).astype('float32')
        tform = np.vstack((tform, hrow))

    try:
        # Run the ECC algorithm. The results are stored in tform
        ccval, tform = cv2.findTransformECC(
                templateImage=img_fixed,
                inputImage=img_moving,
                warpMatrix=tform,
                motionType=cvmotion[motion_type],
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=gaussFiltSize)
    except cv2.error:
        logger.error("failed to align images.")
        raise

    # not all the transforms output 3 x 3
    if tform.shape == (2, 3):
        tform = np.vstack((tform, [0, 0, 1]))

    return tform


def contrast_adjust(image, CLAHE_grid, CLAHE_clip):
    """contrast adjust images

    Parameters
    ----------
    image: numpy.ndaray
        image to pre-process. CV_8UC1 (uint8) or CV_16UC1 (uint16)
    CLAHE_grid : int
        passed as tileGridSize to cv2.createCLAHE. If -1, skipped
    CLAHE_clip : float
        passed as clipLimit to cv2.createCLAHE

    Returns
    -------
    image: numpy.narray
        contrast-adjusted image

    """
    if CLAHE_grid != -1:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_clip,
            tileGridSize=(CLAHE_grid, CLAHE_grid))
        image = clahe.apply(src=image)
    return image


def warp_image(image, tform, motion_type, dst_shape):
    """

    Parameters
    ----------
    image: numpy.ndarray
        image to warp. Assumes uint8
    tform: numpy.ndarray
        3x3 transformation matrix
    motion_type: str
        kind of motion. See function `register_image_pair` for possible
        values
    dst_shape: tuple
        desired shape of warped image

    Returns
    -------
    image: numpy.ndarray
        warped image, forced to uint8

    """
    if motion_type == 'MOTION_HOMOGRAPHY':
        warp = cv2.warpPerspective
    else:
        warp = cv2.warpAffine
        if tform.shape[0] == 3:
            tform = tform[0:2]

    image = warp(
            src=image,
            M=tform,
            dsize=dst_shape[::-1],
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP).astype(np.uint8)

    return image
