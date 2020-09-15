import pytest
import nway.utils as utils
import numpy as np
from jinja2 import Template
import os
import json
import PIL.Image

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


def test_mask_from_json(experiments):
    for exp in experiments:
        mask, mdict = utils.labeled_mask_from_experiment(exp)
        assert len(mask.shape) == 3
        assert (
                set(np.unique(mask[mask != 0])) ==
                set(np.array(list(mdict.keys())).astype(int)))


def test_create_nice_mask(experiments, tmpdir):
    exp = experiments[0]
    output_dir = str(tmpdir.mkdir("nice_mask"))
    nice_exp = utils.create_nice_mask(exp, output_dir)
    for k in exp:
        assert k in nice_exp
        assert exp[k] == nice_exp[k]
    for k in ['nice_mask_path', 'nice_dict_path']:
        assert k in nice_exp
        assert os .path.isfile(nice_exp[k])


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


@pytest.mark.parametrize('depth', [1, 4])
def test_read_tiff_3d(depth, tmpdir):
    mask = np.random.randint(0, 500, size=(depth, 100, 100)).astype('uint16')
    output_dir = str(tmpdir.mkdir('tiffs'))
    fname = os.path.join(output_dir, 'tiff_%d.tif' % depth)

    mask_list = [PIL.Image.fromarray(ix) for ix in mask]
    mask_list[0].save(fname, save_all=True, append_images=mask_list[1:])

    new_mask = utils.read_tiff_3d(fname)

    assert new_mask.shape == mask.shape
    assert np.all(new_mask == mask)


@pytest.mark.parametrize(
        "rois, shape, expected",
        [
            (  # 2 overlapping ROIS
                [
                    {
                        'id': 1,
                        'x': 2,
                        'y': 2,
                        'width': 2,
                        'height': 2,
                        'mask_matrix': [[True, True],
                                        [True, True]]
                        },
                    {
                        'id': 2,
                        'x': 3,
                        'y': 3,
                        'width': 2,
                        'height': 2,
                        'mask_matrix': [[True, True],
                                        [True, True]]
                        }],
                (5, 5),
                np.array([
                    [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0],
                     [0, 0, 1, 1, 0],
                     [0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 2, 2],
                     [0, 0, 0, 2, 2]]]).astype('uint32')),
            (  # 2 non-overlapping ROIs
                [
                    {
                        'id': 1,
                        'x': 1,
                        'y': 1,
                        'width': 2,
                        'height': 2,
                        'mask_matrix': [[True, True],
                                        [True, True]]
                        },
                    {
                        'id': 2,
                        'x': 3,
                        'y': 3,
                        'width': 2,
                        'height': 2,
                        'mask_matrix': [[True, True],
                                        [True, True]]
                        }],
                (5, 5),
                np.array([
                    [[0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 0, 0, 2, 2],
                     [0, 0, 0, 2, 2]]]).astype('uint32')),
            (  # 3 ROIs destined for 2 layers, out of order
                [
                    {
                        'id': 1,
                        'x': 1,
                        'y': 1,
                        'width': 2,
                        'height': 2,
                        'mask_matrix': [[True, True],
                                        [True, True]]
                        },
                    {
                        'id': 2,
                        'x': 2,
                        'y': 2,
                        'width': 2,
                        'height': 2,
                        'mask_matrix': [[True, True],
                                        [True, True]]
                        },
                    {
                        'id': 3,
                        'x': 3,
                        'y': 3,
                        'width': 2,
                        'height': 2,
                        'mask_matrix': [[True, True],
                                        [True, True]]
                        }],
                (5, 5),
                np.array([
                    [[0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 0, 0, 3, 3],
                     [0, 0, 0, 3, 3]],
                    [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 2, 2, 0],
                     [0, 0, 2, 2, 0],
                     [0, 0, 0, 0, 0]]]).astype('uint32')),
            (  # no ROIs
                [],
                (5, 5),
                np.array([
                    [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]]).astype('uint32')),
                    ])
def test_layered_mask_from_rois(rois, shape, expected):
    masks = utils.layered_mask_from_rois(rois, shape)
    np.testing.assert_array_equal(masks, expected)
