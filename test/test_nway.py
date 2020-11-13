import pytest
import nway.nway_matching as nway
import os
from jinja2 import Template
import json
import numpy as np
import copy


TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')


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
    rendered['log_level'] = "DEBUG"
    yield rendered


def test_nway_exception(tmpdir, input_file):
    args = copy.deepcopy(input_file)
    args['experiment_containers']['ophys_experiments'] = \
        [args['experiment_containers']['ophys_experiments'][0]]
    output_dir = str(tmpdir.mkdir("nway_exception"))
    args['output_directory'] = output_dir
    args['assignment_solver'] = 'Blossom'

    nwmatch = nway.NwayMatching(input_data=args, args=[])
    with pytest.raises(nway.NwayException):
        nwmatch.run()


@pytest.mark.parametrize('greduce', ['keepmin', 'popmax'])
def test_default_nway(tmpdir, input_file, greduce):
    args = copy.deepcopy(input_file)
    output_dir = str(tmpdir.mkdir("nway_default"))
    args['output_json'] = os.path.join(output_dir, 'output.json')
    args['pruning_method'] = greduce
    args['include_original'] = True
    n = nway.NwayMatching(input_data=args, args=[])

    assert n.args['assignment_solver'] == 'Blossom'

    n.run()

    with open(n.args['output_json'], 'r') as f:
        oj = json.load(f)

    assert len(oj['nway_matches']) > 650
    nave = np.array([len(i) for i in oj['nway_matches']]).mean()
    assert nave > 1.5

    nexp = len(args['experiment_containers']['ophys_experiments'])

    npairs = int(nexp * (nexp - 1) / 2)

    assert npairs == len(oj['pairwise_results'])
