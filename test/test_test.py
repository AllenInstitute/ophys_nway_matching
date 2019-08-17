import pytest
from nway.nway_matching_main import NwayMatching
import os
from jinja2 import Template
import json
import numpy as np

TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')


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
    input_json = os.path.join(output_dir, 'input.json')
    with open(input_json, 'w') as f:
        json.dump(rendered, f, indent=2)
    yield input_json


@pytest.mark.parametrize(
        "exe",
        [
            #None,
            ("/shared/bioapps/infoapps/lims2_modules/"
             "CAM/ophys_ophys_registration/bp_matching")])
def test_against_old_results(input_file, exe):
    assert os.path.isfile(input_file)
    with open(input_file, 'r') as f:
        j = json.load(f)
    assert os.path.isdir(j['output_directory'])
    for i in j['experiment_containers']['ophys_experiments']:
        assert os.path.isfile(i['ophys_average_intensity_projection_image'])
        assert os.path.isfile(i['max_int_mask_image'])

    args = {}
    args['input_json'] = input_file
    args['output_json'] = os.path.join(
            os.path.dirname(input_file), 'output.json')
    args['munkres_executable'] = exe
    #args['motionType'] = "MOTION_EUCLIDEAN"
    args['log_level'] = 'DEBUG'
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
    k1 = j1['cell_rois'].keys()
    k2 = j2['cell_rois'].keys()
    assert set(k1) == set(k2)
    for k in k1:
        assert np.all(j1['cell_rois'][k] == j2['cell_rois'][k])

    # compare old matching table and new one
    m1 = np.loadtxt(
        os.path.join(os.path.dirname(input_file), "matching_result.txt"))
    m2 = np.loadtxt(
        os.path.join(thistest, "matching_result.txt"))
    assert np.all(m1 == m2)
