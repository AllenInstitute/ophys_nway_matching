import pytest
import pandas as pd
import numpy as np
import os
from jinja2 import Template
import json
import itertools
import nway.utils as utils
import nway.pairwise_matching as pairwise

TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')


@pytest.fixture(scope='function')
def input_file(tmpdir):
    thistest = os.path.join(TEST_FILE_DIR, 'test0')
    myinput = os.path.join(thistest, 'input.json')
    with open(myinput, 'r') as f:
        template = Template(json.dumps(json.load(f)))
    output_dir = str(tmpdir.mkdir("pairwise_test"))
    rendered = json.loads(
            template.render(
                output_dir=output_dir,
                test_files_dir=str(thistest)))
    input_json = os.path.join(output_dir, 'input.json')
    with open(input_json, 'w') as f:
        json.dump(rendered, f, indent=2)
    yield input_json


def read_pairs_from_simple(cost, val):
    inds = np.argwhere(np.array(cost) == val)
    pairs = []
    for i in inds:
        pairs.append([
            cost.index[i[0]],
            cost.columns[i[1]]])
    return pairs


def shuffle_dataframe(df):
    cshuff = list(df.columns)
    np.random.shuffle(cshuff)
    df = df[cshuff]
    df = df.sample(frac=1)
    return df


@pytest.mark.parametrize('shuffle, niter', zip([True, False], [20, 1]))
@pytest.mark.parametrize('solver', ['Hungarian', 'Blossom'])
def test_solvers_easy(solver, shuffle, niter):
    nrow = 50
    ncol = 100
    rows = ['r_%d' % i for i in range(nrow)]
    cols = ['c_%d' % i for i in range(ncol)]
    high_cost = 1000.0
    low_cost = 1.0
    cost_arr = np.ones((nrow, ncol)) * high_cost
    for i in range(nrow):
        cost_arr[i, i] = low_cost

    cost = pd.DataFrame(cost_arr, rows, cols)

    # ground truth for this test
    validation_pairs = read_pairs_from_simple(cost, low_cost)
    validation_set = set([tuple(i) for i in validation_pairs])

    test_sets = []
    for i in range(niter):
        # result should not be order-dependent
        if shuffle:
            cost = shuffle_dataframe(cost)

        pairs = pairwise.gen_assignment_pairs(cost, solver)

        test_sets.append(set([tuple(i) for i in pairs]))

    for test_set in test_sets:
        assert validation_set == test_set


@pytest.mark.parametrize('solver', ['Hungarian', 'Blossom'])
def test_real_cost_data(input_file, tmpdir, solver):
    # Hungarian method only stable under permutations at the ~80% level
    # Blossom stable at the > 95% level
    assert os.path.isfile(input_file)
    with open(input_file, 'r') as f:
        j = json.load(f)

    output_dir = str(tmpdir.mkdir("real_cost"))

    fixed = utils.create_nice_mask(
            j['experiment_containers']['ophys_experiments'][0],
            output_dir)
    moving = utils.create_nice_mask(
            j['experiment_containers']['ophys_experiments'][1],
            output_dir)

    # NOTE specifying solver here only matters for code coverage
    # of input schema, as the cost matrix is used afterwards
    # to calculate pairs.
    pair_args = {
            'fixed': fixed,
            'moving': moving,
            'output_directory': output_dir,
            'assignment_solver': solver,
            'edge_buffer': 0,
            'output_json': os.path.join(
                    output_dir,
                    "{}_to_{}_output.json".format(moving['id'], fixed['id'])),
            'log_level': "DEBUG"
            }

    pm = pairwise.PairwiseMatching(input_data=pair_args, args=[])
    pm.run()

    cost = pm.cost_matrix

    def iou(a, b):
        i = len(a.intersection(b))
        u = len(a.union(b))
        return float(i) / u

    test_sets = []
    for i in range(100):
        cost = shuffle_dataframe(cost)

        pairs = pairwise.gen_assignment_pairs(cost, solver)
        test_sets.append(set([tuple(i) for i in pairs]))

    combos = np.array([
        iou(a, b) for a, b in itertools.combinations(test_sets, 2)])

    if 'Hungarian' in solver:
        assert combos.mean() > 0.75
        assert combos.mean() < 0.85
    else:
        assert combos.mean() > 0.95


def test_gen_matching_table():
    nrow = 10
    ncol = 15
    rows = ['r_%d' % i for i in range(nrow)]
    cols = ['c_%d' % i for i in range(ncol)]
    high_cost = 1000.0
    low_cost = 1.0
    cost_arr = np.ones((nrow, ncol)) * high_cost
    for i in range(nrow):
        cost_arr[i, i] = low_cost
    cost = pd.DataFrame(cost_arr, rows, cols)

    validation_pairs = read_pairs_from_simple(cost, low_cost)

    distance = pd.DataFrame(np.ones(cost_arr.shape) * 1000, rows, cols)
    iou = pd.DataFrame(np.random.randn(*cost_arr.shape), rows, cols)

    for pair in validation_pairs:
        distance[pair[1]][pair[0]] = 1.234
        iou[pair[1]][pair[0]] = 5.678

    # this pair will not be in matches, but in rejected
    # it is below the distance threshold, but it does not
    # show up in the validation_pairs list
    rej_pair = ['r_9', 'c_11']
    distance[rej_pair[1]][rej_pair[0]] = 1.234

    matches, rejected = pairwise.gen_matching_table(
            cost, validation_pairs, distance, iou, 10.0)

    valsets = set([tuple(i) for i in validation_pairs])
    msets = set([(m['fixed'], m['moving']) for m in matches])
    assert valsets == msets
    mdists = np.array([m['distance'] for m in matches])
    assert np.all(np.isclose(mdists, 1.234))
    mious = np.array([m['iou'] for m in matches])
    assert np.all(np.isclose(mious, 5.678))

    rvalsets = set([tuple(i) for i in [rej_pair]])
    rmsets = set([(m['fixed'], m['moving']) for m in rejected])
    assert rvalsets == rmsets
    assert np.isclose(rejected[0]['distance'], 1.234)


def test_region_properties():
    mask = np.zeros((3, 100, 100)).astype('uint16')
    mask[0, 40:50, 40:50] = 1
    i1 = np.argwhere(mask[0] == 1)
    mask[1, 60:65, 30:35] = 2
    i2 = np.argwhere(mask[1] == 2)
    mdict = {
            'mask_dict': {
                "1": 123456,
                "2": 789012,
                }
            }

    prop = pairwise.region_properties(mask, mdict)

    assert prop['centers'].shape == (2, 2)
    ic1 = i1.mean(axis=0)[::-1]
    ic2 = i2.mean(axis=0)[::-1]
    assert np.all(np.isclose(
        prop['centers'],
        np.array([ic1, ic2])))


def test_calc_distance_iou():
    mask1 = np.zeros((3, 100, 100)).astype('uint16')
    mask1[0, 40:50, 40:50] = 1
    mask1[1, 60:65, 30:35] = 2
    dict1 = {
            'mask_dict': {
                "1": 123456,
                "2": 789012,
                }
            }
    mask2 = np.zeros((3, 100, 100)).astype('uint16')
    mask2[0, 38:50, 45:50] = 1
    mask2[1, 63:70, 30:35] = 2
    dict2 = {
            'mask_dict': {
                "1": 987654,
                "2": 321987,
                "3": 100000
                }
            }

    distance, iou = pairwise.calculate_distance_and_iou(
            mask1,
            mask2,
            dict1,
            dict2)

    assert set(distance.columns.tolist()) == set(dict2['mask_dict'].values())
    assert set(distance.index.tolist()) == set(dict1['mask_dict'].values())
    assert set(iou.columns.tolist()) == set(dict2['mask_dict'].values())
    assert set(iou.index.tolist()) == set(dict1['mask_dict'].values())


def test_calculate_cost_matrix():
    nrow = 10
    ncol = 15
    rows = ['r_%d' % i for i in range(nrow)]
    cols = ['c_%d' % i for i in range(ncol)]
    darr = np.random.rand(nrow, ncol) * 100
    distance = pd.DataFrame(darr, rows, cols)
    iarr = np.ones((nrow, ncol)) * 0.5
    iou = pd.DataFrame(iarr, rows, cols)

    cost = pairwise.calculate_cost_matrix(distance, iou, 10)
    assert np.all(cost.columns.tolist() == distance.columns.tolist())
    assert np.all(cost.index.tolist() == distance.index.tolist())

    carr = np.array(cost)

    ind = darr > 10
    assert np.all(carr[ind] == 999.5)
    ind = darr <= 10
    assert np.all(carr[ind] == (darr[ind]/10 + 0.5))
