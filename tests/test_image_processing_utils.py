import nway.image_processing_utils as imu
import pytest
from pathlib import Path
import numpy as np
import PIL.Image

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
