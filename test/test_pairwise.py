import pytest
import pandas as pd
import numpy as np
import os
from jinja2 import Template
import json
import itertools
import nway.utils as utils
from nway.pairwise_matching import gen_assignment_pairs, PairwiseMatching

TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')

cppexe = ("/shared/bioapps/infoapps/lims2_modules/"
          "CAM/ophys_ophys_registration/bp_matching")


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
@pytest.mark.parametrize('solver', ['Hungarian', 'Hungarian-cpp', 'Blossom'])
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

        pairs = gen_assignment_pairs(cost, solver, cppexe)

        test_sets.append(set([tuple(i) for i in pairs]))

    for test_set in test_sets:
        assert validation_set == test_set


@pytest.mark.parametrize('solver', ['Hungarian', 'Hungarian-cpp', 'Blossom'])
def test_real_cost_data(input_file, tmpdir, solver):
    # Hungarian method only stable under permutations at the ~80% level
    # Blossom stable at the > 95% level
    assert os.path.isfile(input_file)
    with open(input_file, 'r') as f:
        j = json.load(f)

    output_dir = str(tmpdir.mkdir("real_cost"))

    fixed = utils.create_nice_mask(
            j['experiment_containers']['ophys_experiments'][0],
            output_dir,
            legacy_label_order=True)
    moving = utils.create_nice_mask(
            j['experiment_containers']['ophys_experiments'][1],
            output_dir,
            legacy_label_order=True)

    pair_args = {
            # stability assertions are true even when
            # using floats and IOUs
            'integer_centroids': False,
            'iou_rounding': False,
            'fixed': fixed,
            'moving': moving,
            'hungarian_executable': cppexe,
            'output_directory': output_dir,
            'output_json': os.path.join(
                    output_dir,
                    "{}_to_{}_output.json".format(moving['id'], fixed['id']))
            }

    pm = PairwiseMatching(input_data=pair_args, args=[])
    pm.run()

    cost = pm.cost_matrix

    def iou(a, b):
        i = len(a.intersection(b))
        u = len(a.union(b))
        return float(i) / u

    test_sets = []
    for i in range(100):
        cost = shuffle_dataframe(cost)

        pairs = gen_assignment_pairs(cost, solver, cppexe)
        test_sets.append(set([tuple(i) for i in pairs]))

    combos = np.array([
        iou(a, b) for a, b in itertools.combinations(test_sets, 2)])

    if 'Hungarian' in solver:
        assert combos.mean() > 0.75
        assert combos.mean() < 0.85
    else:
        assert combos.mean() > 0.95
