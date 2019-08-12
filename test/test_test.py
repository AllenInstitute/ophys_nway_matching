import pytest
from nway.nway_matching_main import main, sum_me
import os
from jinja2 import Template
import json

TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')

@pytest.fixture(scope='function')
def input_file(tmpdir):
    basename = '782536745_ophys_cell_matching_input.json'
    myinput = os.path.join(TEST_FILE_DIR, basename)
    with open(myinput, 'r') as f:
        template = Template(json.dumps(json.load(f)))
    output_dir = str(tmpdir.mkdir("nway_test"))
    rendered = json.loads(
            template.render(
                output_dir=output_dir,
                test_files_dir=str(TEST_FILE_DIR)))
    input_json = os.path.join(output_dir, basename)
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
