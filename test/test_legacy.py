import pytest
from nway.nway_matching import NwayMatching
import os
from jinja2 import Template
import json

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
    input_json = os.path.join(output_dir, 'input.json')
    with open(input_json, 'w') as f:
        json.dump(rendered, f, indent=2)
    yield input_json


def test_against_old_results(input_file):
    assert os.path.isfile(input_file)
    with open(input_file, 'r') as f:
        args = json.load(f)
    assert os.path.isdir(args['output_directory'])
    for i in args['experiment_containers']['ophys_experiments']:
        assert os.path.isfile(i['ophys_average_intensity_projection_image'])
        assert os.path.isfile(i['max_int_mask_image'])

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

    oldset = set([tuple(i) for i in j1['cell_rois'].values()])
    newset = set([tuple(i) for i in j2['nway_matches']])

    def iou(a, b):
        i = len(a.intersection(b))
        u = len(a.union(b))
        return float(i) / u

    # NOTE upgrading from python 2.7 to 3.7 and opencv to 4.1 from
    # 3.4.0 made exact legacy result reproduction impossible
    # a relaxed comparison:
    assert iou(oldset, newset) > 0.82
