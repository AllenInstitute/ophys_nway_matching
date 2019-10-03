import pytest
from nway.nway_matching import NwayMatching
import os
from jinja2 import Template
import json
import numpy as np

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
    output_dir = str(tmpdir.mkdir("nway_test"))
    rendered = json.loads(
            template.render(
                output_dir=output_dir,
                test_files_dir=str(thistest)))
    input_data_json = os.path.join(output_dir, 'input.json')
    with open(input_data_json, 'w') as f:
        json.dump(rendered, f, indent=2)
    yield input_data_json


def test_against_old_results(input_file):
    assert os.path.isfile(input_file)
    with open(input_file, 'r') as f:
        j = json.load(f)
    assert os.path.isdir(j['output_directory'])
    for i in j['experiment_containers']['ophys_experiments']:
        assert os.path.isfile(i['ophys_average_intensity_projection_image'])
        assert os.path.isfile(i['max_int_mask_image'])

    args = {}
    args['input_data_json'] = input_file
    args['output_json'] = os.path.join(
            os.path.dirname(input_file), 'output.json')
    args['legacy'] = True
    args['hungarian_executable'] = cppexe
    args['save_pairwise_tables'] = True
    n = NwayMatching(input_data=args, args=[])
    n.run()

    # compare old output json and new one
    thistest = os.path.join(TEST_FILE_DIR, 'test0')
    old_output_json = os.path.join(thistest, 'output.json')
    new_output_json = os.path.join(
            os.path.dirname(input_file), 'output.json')

    with open(old_output_json, 'r') as f:
        j1 = json.load(f)
    with open(new_output_json, 'r') as f:
        j2 = json.load(f)

    old_in_new = np.zeros(len(j1['cell_rois']))
    for i, old in enumerate(j1['cell_rois'].values()):
        for new in j2['nway_matches']:
            if set(old) == set(new):
                old_in_new[i] += 1

    new_in_old = np.zeros(len(j2['nway_matches']))
    for i, new in enumerate(j2['nway_matches']):
        for old in j1['cell_rois'].values():
            if set(old) == set(new):
                new_in_old[i] += 1

    assert np.count_nonzero(old_in_new == 1) == len(j1['cell_rois'])
    assert np.count_nonzero(new_in_old == 1) == len(j2['nway_matches'])
