import pytest
import nway.utils as utils
import numpy as np
from jinja2 import Template
import os
import json

TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')


@pytest.fixture(scope='function')
def experiments(tmpdir):
    thistest = os.path.join(TEST_FILE_DIR, 'test0')
    myinput = os.path.join(thistest, 'input.json')
    with open(myinput, 'r') as f:
        template = Template(json.dumps(json.load(f)))
    output_dir = str(tmpdir.mkdir("nway_test"))
    rendered = json.loads(
            template.render(
                output_dir=output_dir,
                test_files_dir=str(thistest)))
    yield rendered['experiment_containers']['ophys_experiments']


@pytest.mark.parametrize("legacy", [True, False])
def test_mask_from_json(experiments, legacy):
    for exp in experiments:
        mask, mdict = utils.labeled_mask_from_experiment(exp, legacy=legacy)
        assert len(mask.shape) == 3
        assert (
                set(np.unique(mask[mask != 0])) ==
                set(np.array(mdict.keys()).astype(int)))


@pytest.mark.parametrize("legacy", [True, False])
def test_relabel(experiments, legacy):
    for exp in experiments:
        mask, _ = utils.labeled_mask_from_experiment(exp, legacy=legacy)
        remask = utils.relabel(mask)
        umask = np.unique(remask)
        assert np.unique(mask).size == umask.size
        assert mask.shape == remask.shape
        if legacy:
            assert np.all(umask == np.arange(umask.size))


def test_too_many_labels():
    mask = np.zeros(50 * 1000 * 1000)
    inds = np.random.choice(np.arange(mask.size), int(mask.size * 0.05))
    mask[inds] = 1
    mask = mask.reshape(50, 1000, 1000)
    with pytest.raises(AssertionError):
        remask = utils.relabel(mask)


def test_row_col():
    h = 120
    w = 100
    roi = {
            "mask_matrix": np.random.randint(
                0, 2, size=(h, w)).astype(bool),
            "x": np.random.randint(1, 1000),
            "y": np.random.randint(1, 1000),
            "height": h,
            "width": w
            }
    inds = []
    for ix in range(w):
        for iy in range(h):
            if roi['mask_matrix'][iy, ix]:
                inds.append([
                    iy + roi["y"],
                    ix + roi["x"]])
    inds = np.array(inds)
    masked = utils.row_col_from_roi(roi)
    isets = set([tuple(i) for i in inds])
    msets = set([tuple(i) for i in masked])
    assert msets == isets


def test_first_order_properties():
    M = np.eye(3, 3)
    M[0:2, 2] = np.random.randn(2)
    prop = utils.calc_first_order_properties(M)
    assert prop['scale'] == (1.0, 1.0)
    assert np.isclose(prop['shear'], 0.0)
    assert np.all(prop['translation'] == M[0: 2, 2])
    assert np.isclose(prop['rotation'], 0.0)
    assert 'warning' not in prop

    M[2, 1] = 0.1
    prop = utils.calc_first_order_properties(M)
    assert prop['scale'] == (1.0, 1.0)
    assert np.isclose(prop['shear'], 0.0)
    assert np.all(prop['translation'] == M[0: 2, 2])
    assert np.isclose(prop['rotation'], 0.0)
    assert 'warning' in prop
