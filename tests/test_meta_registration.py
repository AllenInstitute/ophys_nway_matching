import pytest
import numpy as np
import nway.meta_registration as mr
import nway.transforms as tf


@pytest.mark.parametrize("include_original", [True, False])
def test_MetaRegistration_init(include_original):
    m = mr.MetaRegistration(include_original=include_original)
    names = []
    for candidate in m.candidates:
        assert isinstance(candidate, tf.TransformList)
        names.append([i.__class__.__name__
                      for i in candidate.transforms])
    if include_original:
        assert ['CLAHE', 'ECC'] in names
    if not include_original:
        assert ['CLAHE', 'ECC'] not in names

    assert isinstance(m.contrast, tf.CLAHE)
    assert callable(m.measure)


def test_MetaRegistration_estimate():
    candidates = [
            tf.TransformList(
                transforms=[tf.Crop(edge_buffer=1), tf.Transform()]),
            tf.TransformList(
                transforms=[tf.Crop(edge_buffer=257), tf.Transform()]),
            tf.TransformList(transforms=[tf.Transform(), tf.Transform()])]
    src = np.ones((512, 512)).astype('uint8')
    dst = np.ones((512, 512)).astype('uint8')

    # transforms do not have matrix attribute before estmation
    for candidate in candidates:
        for tform in candidate.transforms:
            assert not hasattr(tform, 'matrix')
    candidates, failed = mr.MetaRegistration.estimate(candidates, src, dst)

    # intentionally made one raise an exception (the large crop)
    assert True in failed

    # successful estimations now have matrix attribute
    for fail, candidate in zip(failed, candidates):
        if not fail:
            for tform in candidate.transforms:
                assert hasattr(tform, 'matrix')


def test_MetaRegistration_evaluate():
    candidates = [
            tf.TransformList(
                transforms=[tf.Crop(edge_buffer=1), tf.Transform()]),
            tf.TransformList(
                transforms=[tf.Crop(edge_buffer=257), tf.Transform()]),
            tf.TransformList(transforms=[tf.Transform(), tf.Transform()])]
    src = np.ones((512, 512)).astype('uint8')
    dst = np.ones((512, 512)).astype('uint8')
    candidates, failed = mr.MetaRegistration.estimate(candidates, src, dst)

    def measure(src, dst):
        return 1.0
    contrast = tf.CLAHE(CLAHE_grid=24, CLAHE_clip=2.5)

    scores = mr.MetaRegistration.evaluate(candidates, failed,
                                          'MOTION_EUCLIDEAN',
                                          contrast, measure, src, dst)
    assert len(scores) == len(candidates)
    for fail, score in zip(failed, scores):
        if fail:
            assert score is None
        else:
            assert score == 1.0


@pytest.mark.parametrize(
        "scores, expected",
        [
            ([0.1, 0.2, 0.3, 0.14], 2),
            ([0.1, 0.2, None, 0.14], 1)])
def test_MetaRegistration_select(scores, expected):
    index = mr.MetaRegistration.select(scores)
    assert index == expected


def test_MetaRegistration_call():
    src = np.ones((512, 512)).astype('uint8')
    dst = np.ones((512, 512)).astype('uint8')

    m = mr.MetaRegistration()
    m(src, dst)
    for att in ['failed', 'scores', 'best_matrix', 'best_candidate']:
        assert hasattr(m, att)
