import pytest
from nway.nway_matching_main import sum_me, NwayMatching
import os
from jinja2 import Template
import json

TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')

@pytest.fixture(scope='function')
def input_file(tmpdir):
    basename = '782536745_ophys_cell_matching_input.json'
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


def test_first_test():
    assert sum_me(2, 2) == 4


def test_temp(input_file):
    assert os.path.isfile(input_file)
    with open(input_file, 'r') as f:
        j = json.load(f)
    assert os.path.isdir(j['output_directory'])
    for i in j['experiment_containers']['ophys_experiments']:
        assert os.path.isfile(i['ophys_average_intensity_projection_image'])
        assert os.path.isfile(i['max_int_mask_image'])

    args = {}
    args['input_json'] = input_file
    args['munkres_executable'] = '/shared/bioapps/infoapps/lims2_modules/CAM/ophys_ophys_registration/bp_matching'
    args['output_json'] = os.path.join(os.path.dirname(input_file), 'output.json')
    n = NwayMatching(input_data=args, args=[])
    n.run()
