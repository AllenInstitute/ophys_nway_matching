import cv2
import numpy as np


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
