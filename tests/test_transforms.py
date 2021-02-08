import pytest
import contextlib
import cv2
import PIL.Image
import numpy as np
from pathlib import Path
import nway.transforms as nwtf
import nway.image_processing_utils as impu

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
def moving_image_fixture(fixed_image_fixture, request):
    immoving = cv2.warpPerspective(
            fixed_image_fixture,
            request.param.get("transform"),
            fixed_image_fixture.shape[::-1])
    return immoving, request.param


@pytest.mark.parametrize(
        "src, dst",
        [
            (
                np.uint8([[1, 2, 3, 4],
                          [4, 1, 2, 3],
                          [3, 4, 1, 2],
                          [2, 3, 4, 1]]),
                np.uint8([[5, 6, 7, 8],
                          [8, 5, 6, 7],
                          [7, 8, 5, 6],
                          [6, 7, 8, 5]]))])
def test_Transform(src, dst):
    tf = nwtf.Transform()
    tf.estimate(src, dst)
    np.testing.assert_array_equal(tf.matrix, np.eye(3).astype('float32'))
    mov, fix = tf.apply(src, dst)
    np.testing.assert_array_equal(src, mov)
    np.testing.assert_array_equal(fix, dst)


@pytest.mark.parametrize(
        "src, dst, grid, clip",
        [
            (
                np.random.randint(0, 255, dtype='uint8', size=(482, 512)),
                np.random.randint(0, 255, dtype='uint8', size=(482, 512)),
                24, 2.5
                )])
def test_CLAHE(src, dst, grid, clip):
    tf = nwtf.CLAHE(CLAHE_grid=grid, CLAHE_clip=clip)
    assert tf.CLAHE_grid == grid
    assert tf.CLAHE_clip == clip
    # no geometrical transform here, should inherit identity
    tf.estimate(src, dst)
    np.testing.assert_array_equal(tf.matrix, np.eye(3).astype('float32'))
    # just check shapes
    mov, fix = tf.apply(src, dst)
    assert mov.shape == src.shape
    assert fix.shape == dst.shape


@pytest.mark.parametrize(
        "src, expected",
        [
            (
                np.uint8([[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 4, 4, 4, 0, 0, 0],
                          [0, 0, 8, 9, 8, 0, 0, 0],
                          [0, 0, 3, 3, 3, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]]),
                np.float32([[1.0, 0.0, -1.0],
                            [0.0, 1.0, 2.0],
                            [0.0, 0.0, 1.0]]))])
def test_PhaseCorrelate(src, expected):
    dst = impu.warp_image(src, expected, "MOTION_EUCLIDEAN", src.shape)
    tf = nwtf.PhaseCorrelate()
    tf.estimate(src, dst)
    mov, fix = tf.apply(src, dst)
    np.testing.assert_array_equal(fix, dst)
    np.testing.assert_array_equal(mov, dst)
    np.testing.assert_allclose(tf.matrix, expected, rtol=1e-3)


@pytest.mark.parametrize(
        "src, dst, edge, expected_src, expected_dst, context",
        [
            (
                np.uint8([[1, 2, 3, 4],
                          [0, 1, 2, 3],
                          [2, 3, 4, 4],
                          [2, 3, 6, 7]]),
                np.uint8([[1, 2, 3, 4],
                          [0, 5, 7, 3],
                          [2, 5, 8, 4],
                          [2, 3, 6, 7]]),
                1,
                np.uint8([[1, 2],
                          [3, 4]]),
                np.uint8([[5, 7],
                          [5, 8]]),
                contextlib.nullcontext()),
            (
                np.uint8([[1, 2, 3, 4],
                          [0, 1, 2, 3],
                          [2, 3, 4, 4],
                          [2, 3, 6, 7]]),
                np.uint8([[1, 2, 3, 4],
                          [0, 5, 7, 3],
                          [2, 5, 8, 4],
                          [2, 3, 6, 7]]),
                0,
                np.uint8([[1, 2, 3, 4],
                          [0, 1, 2, 3],
                          [2, 3, 4, 4],
                          [2, 3, 6, 7]]),
                np.uint8([[1, 2, 3, 4],
                          [0, 5, 7, 3],
                          [2, 5, 8, 4],
                          [2, 3, 6, 7]]),
                contextlib.nullcontext()),
            (
                np.uint8([[1, 2, 3, 4],
                          [0, 1, 2, 3],
                          [2, 3, 4, 4],
                          [2, 3, 6, 7]]),
                np.uint8([[1, 2, 3, 4],
                          [0, 5, 7, 3],
                          [2, 5, 8, 4],
                          [2, 3, 6, 7]]),
                3,
                np.uint8([[1, 2, 3, 4],
                          [0, 1, 2, 3],
                          [2, 3, 4, 4],
                          [2, 3, 6, 7]]),
                np.uint8([[1, 2, 3, 4],
                          [0, 5, 7, 3],
                          [2, 5, 8, 4],
                          [2, 3, 6, 7]]),
                pytest.raises(nwtf.TransformException))
                ])
def test_Crop(src, dst, edge, expected_src, expected_dst, context):
    tf = nwtf.Crop(edge_buffer=edge)
    tf.estimate(src, dst)
    with context:
        csrc, cdst = tf.apply(src, dst)
        np.testing.assert_array_equal(expected_src, csrc)
        np.testing.assert_array_equal(expected_dst, cdst)


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
def test_ECC(fixed_image_fixture, moving_image_fixture,
             motion_type, tform_tol, trans_tol):
    moving_image, param = moving_image_fixture
    expected_transform = param["transform"]

    tf = nwtf.ECC(maxCount=500, epsilon=1e-3, motion_type=motion_type,
                  gaussFiltSize=5)
    tf.estimate(moving_image, fixed_image_fixture)
    transform = tf.matrix

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

        mov, _ = tf.apply(moving_image, fixed_image_fixture)
        np.testing.assert_allclose(
                mov.astype(int),
                fixed_image_fixture.astype(int),
                atol=26)


def test_TransformList():
    # chain together 2 identities
    tf1 = nwtf.Transform()
    tf2 = nwtf.Transform()
    tf_list = nwtf.TransformList(transforms=[tf1, tf2])

    src = np.random.randint(0, 255, dtype='uint8', size=(512, 512))
    dst = np.random.randint(0, 255, dtype='uint8', size=(512, 512))

    tf_list.estimate(src, dst)
    np.testing.assert_array_equal(tf_list.matrix, np.eye(3).astype('float32'))

    mov, fix = tf_list.apply(src, dst)
    np.testing.assert_array_equal(src, mov)
    np.testing.assert_array_equal(fix, dst)
