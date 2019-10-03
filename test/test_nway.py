import pytest
import nway.nway_matching as nway
import os
from jinja2 import Template
import json
from marshmallow import ValidationError
import numpy as np
import itertools
from functools import partial


TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')

cppexe = ("/shared/bioapps/infoapps/lims2_modules/"
          "CAM/ophys_ophys_registration/bp_matching")


@pytest.fixture(scope='module')
def table_to_prune():
    fname = os.path.join(
            TEST_FILE_DIR,
            "test0",
            "table_before_pruning.json")
    with open(fname, 'r') as f:
        j = json.load(f)
    table = np.array(j['table'])
    score = np.array(j['score'])
    yield (table, score)


@pytest.fixture(scope='function')
def input_file(tmpdir):
    thistest = os.path.join(TEST_FILE_DIR, 'test0')
    myinput = os.path.join(thistest, 'input.json')
    with open(myinput, 'r') as f:
        template = Template(json.dumps(json.load(f)))
    output_dir = str(tmpdir.mkdir("nway_test"))
    rendered = json.loads(
            template.render(
                output_dir=output_dir,
                test_files_dir=str(thistest)))
    input_data_json = os.path.join(output_dir, 'input.json')
    with open(input_data_json, 'w') as f:
        json.dump(rendered, f, indent=2)
    yield input_data_json


def test_Nway_legacy_settings(input_file, tmpdir):
    output_dir = str(tmpdir.mkdir("nway_legacy_settings"))
    args = {
            'input_data_json': input_file,
            'output_directory': output_dir,
            'assignment_solver': 'Blossom',
            'legacy': True
            }
    with pytest.raises(ValidationError):
        nwmatch = nway.NwayMatching(input_data=dict(args), args=[])

    args['hungarian_executable'] = cppexe
    nwmatch = nway.NwayMatching(input_data=dict(args), args=[])

    assert nwmatch.args['legacy_label_order']
    assert nwmatch.args['legacy_pruning_index_error']
    assert nwmatch.args['legacy_pruning_order_dependence']
    assert nwmatch.args['integer_centroids']
    assert nwmatch.args['iou_rounding']
    assert nwmatch.args['assignment_solver'] == 'Hungarian-cpp'

    args['legacy'] = False
    nwmatch = nway.NwayMatching(input_data=dict(args), args=[])

    assert not nwmatch.args['legacy_label_order']
    assert not nwmatch.args['legacy_pruning_index_error']
    assert not nwmatch.args['legacy_pruning_order_dependence']
    assert not nwmatch.args['integer_centroids']
    assert not nwmatch.args['iou_rounding']
    assert nwmatch.args['assignment_solver'] == 'Blossom'


def test_nway_exception(tmpdir, input_file):
    with open(input_file, 'r') as f:
        j = json.load(f)
    j['experiment_containers']['ophys_experiments'] = \
        [j['experiment_containers']['ophys_experiments'][0]]
    output_dir = str(tmpdir.mkdir("nway_exception"))
    tmpinput = os.path.join(output_dir, 'tmp.json')
    with open(tmpinput, 'w') as f:
        json.dump(j, f)
    args = {
            'input_data_json': tmpinput,
            'output_directory': output_dir,
            'assignment_solver': 'Blossom',
            }

    nwmatch = nway.NwayMatching(input_data=args, args=[])
    with pytest.raises(nway.NwayException):
        nwmatch.run()


def test_default_nway(input_file):
    args = {}
    args['input_data_json'] = input_file
    args['output_json'] = os.path.join(
            os.path.dirname(input_file), 'output.json')
    n = nway.NwayMatching(input_data=args, args=[])

    assert n.args['assignment_solver'] == 'Blossom'

    n.run()

    with open(n.args['output_json'], 'r') as f:
        oj = json.load(f)

    assert len(oj['nway_matches']) > 650
    nave = np.array([len(i) for i in oj['nway_matches']]).mean()
    assert nave > 1.5

    with open(input_file, 'r') as f:
        inj = json.load(f)
    nexp = len(inj['experiment_containers']['ophys_experiments'])

    npairs = int(nexp * (nexp - 1) / 2)

    assert npairs == len(oj['pairwise_results'])


def shuffle(table, score):
    ind = np.arange(len(table))
    np.random.shuffle(ind)
    ntable = [table[i] for i in ind]
    return ntable, score[ind]


def get_sets(table, old=False):
    x = []
    for t in table:
        x.append(tuple([i for i in t if i != -1]))
    return set(x)


def iou(set1, set2):
    i = float(len(set1.intersection(set2)))
    u = len(set1.union(set2))
    return i / u


@pytest.mark.parametrize(
        'legacy, method',
        [(True, None), (False, "keepmin"), (False, "popmax")])
def test_pruning_legacy(table_to_prune, legacy, method):
    # demonstration that legacy pruning is unstable under
    # permutations
    if legacy:
        fprune = nway.prune_matching_table_legacy
    else:
        fprune = partial(nway.prune_matching_table, method=method)

    table, score = table_to_prune
    sets = []
    ntrials = 5
    for i in range(ntrials):
        table, score = shuffle(table, score)
        pruned = fprune(table, score)
        sets.append(get_sets(pruned))

    ious = np.array([iou(a, b) for a, b in itertools.combinations(sets, 2)])
    if legacy:
        assert (0.8 < ious.mean() < 0.9)
    else:
        assert np.all(ious == 1.0)
