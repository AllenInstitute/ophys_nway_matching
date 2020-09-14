import nway.image_processing_utils as imu
import pytest
from pathlib import Path
import numpy as np
import PIL.Image
import cv2

TEST_FILE_DIR = Path.resolve(Path(__file__)).parent / "test_files"


@pytest.fixture
def fixed_image_fixture():
    # grab one specific file because failing large translation
    # cases vary image-by-image
    fgen = TEST_FILE_DIR.rglob("avgInt_a1X.png")
    for f in fgen:
        if "ophys_session_775289198" in str(f):
            impath = f
    with PIL.Image.open(impath) as fp:
        image = np.array(fp)
    return image


@pytest.fixture
def flipped_image_fixture(fixed_image_fixture):
    flipped = np.flipud(np.fliplr(fixed_image_fixture))
    return flipped


@pytest.fixture
def moving_image_fixture(fixed_image_fixture, request):
    immoving = cv2.warpPerspective(
            fixed_image_fixture,
            request.param.get("transform"),
            fixed_image_fixture.shape[::-1])
    return immoving, request.param


@pytest.mark.parametrize("preregister", [True, False])
@pytest.mark.parametrize("motion_type, tform_tol, trans_tol", [
    ("MOTION_TRANSLATION", (0.0001, 0.0001), (0.5, 0.0)),
    ("MOTION_EUCLIDEAN", (0.0002, 0.0001), (0.5, 0.0)),
    ("MOTION_AFFINE", (0.0005, 0.005), (0.5, 0.0)),
    ("MOTION_HOMOGRAPHY", (0.005, 0.0001), (0.5, 0.0)),
    ])
@pytest.mark.parametrize(
        "moving_image_fixture",
        [
           # identity
           {"transform": np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])},
           # x translation
           {"transform": np.array([[1.0, 0.0, 10.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])},
           # x and y translation
           {"transform": np.array([[1.0, 0.0, 10.0],
                                   [0.0, 1.0, -23.0],
                                   [0.0, 0.0, 1.0]])},
           # 5 deg rotation and x and y translation
           {"transform": np.array([[np.cos(0.087), -np.sin(0.087), 10.0],
                                   [np.sin(0.087), np.cos(0.087), -23.0],
                                   [0.0, 0.0, 1.0]])},
           ],
        indirect=["moving_image_fixture"])
def test_register_image_pair(fixed_image_fixture, moving_image_fixture,
                             preregister, motion_type, tform_tol, trans_tol):
    """
    warp an image and see if the register image can recover the transform.
    NOTE: in production, we are using MOTION_EUCLIDEAN as that is the
    expected type of misalignment from the microscope rigs. The underlying
    opencv functions support affine (includes scale) and homography/perspective
    so those options are exposed in case of a new rig and some experimentation.

    tform_tol is (abs, rel) tolerance on 2x2 multiplicative part
    trans_tol is (abs, rel) tolerance of translations
    (abs, rel) are as specified in np.testing.assert_allclose
    As the order of desired transform goes up, it has more trouble
    matching tolerance
    """
    moving_image, param = moving_image_fixture
    expected_transform = param["transform"]
    transform = imu.register_image_pair(
            fixed_image_fixture,
            moving_image,
            500,
            1e-3,
            motion_type,
            5,
            preregister)

    if not np.allclose(expected_transform[0:2, 0:2], np.eye(2)) & \
            (motion_type == 'MOTION_TRANSLATION'):
        # translation can't be expected to solve for rotation
        pass
    else:
        # the multiplicative part of the transform
        np.testing.assert_allclose(transform[0:2, 0:2],
                                   expected_transform[0:2, 0:2],
                                   atol=tform_tol[0], rtol=tform_tol[1])
        # the translation part of the transform
        np.testing.assert_allclose(transform[0:2, 2],
                                   expected_transform[0:2, 2],
                                   atol=trans_tol[0], rtol=trans_tol[1])


@pytest.mark.parametrize("preregister", [True, False])
@pytest.mark.parametrize("motion_type, tform_tol, trans_tol", [
    ("MOTION_EUCLIDEAN", (0.1, 0.1), (0.5, 0.0)),
    ])
@pytest.mark.parametrize(
        "moving_image_fixture, expect_success",
        [
            # x and y translation
            ({"transform": np.array([[1.0, 0.0, 110.0],
                                    [0.0, 1.0, -123.0],
                                    [0.0, 0.0, 1.0]])}, True),
            # small rotation
            ({"transform": np.array([[np.cos(0.025), -np.sin(0.025), 110.0],
                                    [np.sin(0.025), np.cos(0.025), -123.0],
                                    [0.0, 0.0, 1.0]])}, True),
            # the routine can't handle large rotations and translations
            ({"transform": np.array([[np.cos(0.09), -np.sin(0.09), 110.0],
                                    [np.sin(0.09), np.cos(0.09), -123.0],
                                    [0.0, 0.0, 1.0]])}, False),
            ],
        indirect=["moving_image_fixture"])
def test_preregister_for_big_translations(fixed_image_fixture,
                                          moving_image_fixture, preregister,
                                          motion_type, tform_tol,
                                          trans_tol, expect_success):
    """ the preregistration fixes cases that fail for large offsets. These
    are examples of those cases.
    """
    moving_image, param = moving_image_fixture
    expected_transform = param["transform"]
    transform = imu.register_image_pair(
            fixed_image_fixture,
            moving_image,
            500,
            1e-3,
            motion_type,
            5,
            preregister)

    if preregister & expect_success:
        np.testing.assert_allclose(transform[0:2, 0:2],
                                   expected_transform[0:2, 0:2],
                                   atol=tform_tol[0], rtol=tform_tol[1])
        np.testing.assert_allclose(transform[0:2, 2],
                                   expected_transform[0:2, 2],
                                   atol=trans_tol[0], rtol=trans_tol[1])
    else:
        # failure mode is typically a very small translation solution
        assert not np.allclose(transform[0:2, 2],
                               expected_transform[0:2, 2],
                               atol=trans_tol[0], rtol=trans_tol[1])


def test_failure_to_align(fixed_image_fixture, flipped_image_fixture):
    with pytest.raises(
            cv2.error,
            match=(".*Images may be uncorrelated or non-overlapped "
                   "in function 'findTransformECC")):
        imu.register_image_pair(
                fixed_image_fixture,
                flipped_image_fixture,
                50,
                1e-7,
                'MOTION_EUCLIDEAN',
                5,
                True)


@pytest.mark.parametrize("CLAHE_GRID, CLAHE_CLIP",
                         [(8, 2.5), (24, 2.5), (-1, 2.5)])
def test_contrast_adjust(fixed_image_fixture, CLAHE_GRID, CLAHE_CLIP):
    ca_image = imu.contrast_adjust(fixed_image_fixture, CLAHE_GRID, CLAHE_CLIP)
    assert ca_image.shape == fixed_image_fixture.shape
    assert ca_image.dtype == fixed_image_fixture.dtype


@pytest.mark.parametrize("motion_type", [
                         "MOTION_TRANSLATION", "MOTION_EUCLIDEAN",
                         "MOTION_AFFINE", "MOTION_HOMOGRAPHY"])
def test_warp_image(fixed_image_fixture, motion_type):
    tform = np.array([[1.1, 0.1, 12.3],
                      [-0.5, 0.94, -23],
                      [0.0, 0.0, 1.0]])
    warped = imu.warp_image(
            fixed_image_fixture, tform, motion_type, fixed_image_fixture.shape)
    assert warped.shape == fixed_image_fixture.shape
    assert warped.dtype == 'uint8'
