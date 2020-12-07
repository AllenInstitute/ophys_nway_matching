import json
import os
import shutil
import time

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import nway.diagnostics as nwdi
import PIL.Image
import PyPDF2
import pytest

import pandas as pd

matplotlib.use('Agg')


TEST_FILE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files')


@pytest.fixture(scope='module')
def new_output():
    ojpath = os.path.join(
            TEST_FILE_DIR,
            "test0",
            "new_output.json")
    yield ojpath


def test_cell_lookup(new_output):
    cell_lookup = nwdi.cell_experiment_dict(new_output)
    cells = list(cell_lookup.keys())
    with open(new_output, 'r') as f:
        outj = json.load(f)
    for match in outj['nway_matches']:
        for cell_id in match:
            assert cell_id in cells


def test_pairwise_transforms(new_output):
    ptf = nwdi.pairwise_transforms(new_output)
    assert type(ptf) == dict

    fig = plt.figure(1)
    ptf = nwdi.pairwise_transforms(new_output, fig=fig)
    assert type(ptf) == dict

    outer_plot_grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    subplot_spec = outer_plot_grid[1]
    fig.clf()
    ptf = nwdi.pairwise_transforms(
            new_output, fig=fig, subplot_spec=subplot_spec)
    assert type(ptf) == dict


def test_some_grid():
    for n in range(1, 20):
        (nr, nc) = nwdi.some_grid(n)
        assert (nr * nc) >= n


def test_pairwise_matches(new_output):
    costs, allcosts = nwdi.pairwise_matches(new_output)
    assert type(costs) == dict
    assert type(allcosts) == dict

    fig = plt.figure(1)
    costs, allcosts = nwdi.pairwise_matches(new_output, fig=fig)
    assert type(costs) == dict
    assert type(allcosts) == dict

    outer_plot_grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    subplot_spec = outer_plot_grid[1]
    fig.clf()
    costs, allcosts = nwdi.pairwise_matches(
            new_output, fig=fig, subplot_spec=subplot_spec)
    assert type(costs) == dict
    assert type(allcosts) == dict


def test_nway_matches(new_output):
    allnw = nwdi.nway_matches(new_output)
    assert type(allnw) == dict

    fig = plt.figure(1)
    allnw = nwdi.nway_matches(new_output, fig=fig)
    assert type(allnw) == dict


def test_plot_all(new_output, tmpdir):
    nwdi.plot_all(new_output)

    output_dir = str(tmpdir.mkdir("diagnostic_test"))
    fname = os.path.join(
            output_dir,
            "output.pdf")

    nwdi.plot_all(new_output, fname)
    assert os.path.isfile(fname)
    PyPDF2.PdfFileReader(fname)


def test_NwayDiagnostics(new_output, tmpdir):
    output_dir = str(tmpdir.mkdir("diagnostic_test"))
    fname = os.path.join(
            output_dir,
            "output2.pdf")

    args = {
            'input_json': new_output,
            'output_pdf': fname,
            'log_level': "DEBUG"
            }

    nd = nwdi.NwayDiagnostics(input_data=args, args=[])
    nd.run()
    assert os.path.isfile(fname)
    PyPDF2.PdfFileReader(fname)

    ninput = os.path.join(
            output_dir,
            "tmpinput.json")

    fname3 = os.path.join(
            output_dir,
            "output3.pdf")

    shutil.copy(new_output, ninput)
    args = {
            'input_json': ninput,
            'output_pdf': fname3,
            'use_input_dir': True,
            'log_level': "DEBUG"
            }
    nd = nwdi.NwayDiagnostics(input_data=args, args=[])
    nd.run()

    oname = os.path.join(
            output_dir,
            os.path.basename(args['output_pdf']))
    assert os.path.isfile(oname)
    PyPDF2.PdfFileReader(oname)


@pytest.fixture
def nway_input_fixture(request, tmp_path):
    ophys_experiments = request.param

    expected_id_avg_image_map = {}
    for indx, expt in enumerate(ophys_experiments):
        timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        save_fname = tmp_path / f"{indx}_{timestamp}.png"

        image_array = expt["dummy_avg_image"]
        expected_id_avg_image_map[expt['id']] = image_array

        with PIL.Image.fromarray(image_array) as im:
            im.save(save_fname)
        expt["ophys_average_intensity_projection_image"] = str(save_fname)

    nway_input_dict = {
        "experiment_containers": {
            "ophys_experiments": ophys_experiments
        }
    }
    return (nway_input_dict, expected_id_avg_image_map)


@pytest.mark.parametrize("nway_input_fixture, expected_id_stim_name_map", [
    # Test case 1
    ([{"id": 12345,
       "stimulus_name": "A",
       "dummy_avg_image": np.random.randint(0, 255, size=(5, 5),
                                            dtype=np.uint8)},
      {"id": 54321,
       "stimulus_name": "B",
       "dummy_avg_image": np.random.randint(0, 255, size=(5, 5),
                                            dtype=np.uint8)}],
     {12345: "A", 54321: "B"}),

    # Test case 2
    ([{"id": 56789,
       "stimulus_name": "B",
       "dummy_avg_image": np.random.randint(0, 255, size=(5, 5),
                                            dtype=np.uint8)},
      {"id": 98765,
       "stimulus_name": "A",
       "dummy_avg_image": np.random.randint(0, 255, size=(5, 5),
                                            dtype=np.uint8)}],
     {98765: "A", 56789: "B"}),

    # Test case 3
    ([{"id": 56789,
       "stimulus_name": None,
       "dummy_avg_image": np.random.randint(0, 255, size=(5, 5),
                                            dtype=np.uint8)},
      {"id": 98765,
       "stimulus_name": "B",
       "dummy_avg_image": np.random.randint(0, 255, size=(5, 5),
                                            dtype=np.uint8)}],
     {98765: "B", 56789: "Unknown Stimulus"})
], indirect=["nway_input_fixture"])
def test_create_nway_input_maps(nway_input_fixture, expected_id_stim_name_map):
    nway_input, expected_expt_id_avg_image_map = nway_input_fixture

    obt = nwdi.create_nway_input_maps(nway_input)

    assert obt[0] == expected_id_stim_name_map
    for k, v in expected_expt_id_avg_image_map.items():
        assert k in obt[1]
        assert np.allclose(v, obt[1][k])


@pytest.mark.parametrize(("nway_output, expt_id_stim_name_map, "
                          "expt_id_avg_image_map, expected_df, "
                          "expected_warped_images_map"), [
    # nway_output
    ({"pairwise_results": [
        {"fixed_experiment": 123,
         "moving_experiment": 321,
         "matches": [1, 2],
         "unmatched": {"fixed": [3], "moving": [4]},
         "transform": {"matrix": np.identity(3),
                       "transform_type": "IDENTITY"}},
        {"fixed_experiment": 123,
         "moving_experiment": 456,
         "matches": [1, 2, 3],
         "unmatched": {"fixed": [], "moving": [4]},
         "transform": {"matrix": np.identity(3),
                       "transform_type": "IDENTITY"}},
        {"fixed_experiment": 456,
         "moving_experiment": 321,
         "matches": [1, 2, 3, 4],
         "unmatched": {"fixed": [], "moving": []},
         "transform": {"matrix": np.identity(3),
                       "transform_type": "IDENTITY"}},
     ]},
     # expt_id_stim_name_map
     {123: "A", 321: "B", 456: "C"},
     # expt_id_avg_image_map
     {123: np.identity(9), 321: np.identity(9) * 2, 456: np.identity(9) * 3},
     # expected_df
     pd.DataFrame({
         "fixed_expt": [123, 123, 456],
         "fixed_expt_stim_name": ["A", "A", "C"],
         "moving_expt": [321, 456, 321],
         "moving_expt_stim_name": ["B", "C", "B"],
         "n_unmatched_fixed": [1, 0, 0],
         "n_unmatched_moving": [1, 1, 0],
         "n_matches": [2, 3, 4],
         "n_total": [4, 4, 4],
         "fraction_matched": [0.5, 3/4, 1]}),
     # expected_warped_images_map
     {"321_to_123": np.identity(9) * 2, "123_to_321": np.identity(9),
      "123_to_456": np.identity(9), "456_to_123": np.identity(9) * 3,
      "456_to_321": np.identity(9) * 3, "321_to_456": np.identity(9) * 2}),
])
def test_create_nway_summary_df(nway_output, expt_id_stim_name_map,
                                expt_id_avg_image_map, expected_df,
                                expected_warped_images_map):
    obt = nwdi.create_nway_summary_df(expt_id_stim_name_map,
                                      expt_id_avg_image_map,
                                      nway_output)

    pd.testing.assert_frame_equal(obt, expected_df)

    assert obt.attrs['expt_id_stim_name_map'] == expt_id_stim_name_map

    obt_warped_images = obt.attrs['warped_images']
    for k, v in expected_warped_images_map.items():
        assert k in obt_warped_images
        assert np.allclose(obt_warped_images[k], v)

    obt_avg_image_map = obt.attrs['expt_id_avg_image_map']
    for k, v in expt_id_avg_image_map.items():
        assert k in obt_avg_image_map
        assert np.allclose(obt_avg_image_map[k], v)


@pytest.fixture
def nway_summary_df_fixture() -> pd.DataFrame:

    random_image_array = np.random.randint(0, 255, size=(25, 25),
                                           dtype=np.uint8)

    summary_df = pd.DataFrame({
         "fixed_expt": [123, 123, 456],
         "fixed_expt_stim_name": ["A", "A", "C"],
         "moving_expt": [321, 456, 321],
         "moving_expt_stim_name": ["B", "C", "B"],
         "n_unmatched_fixed": [1, 0, 0],
         "n_unmatched_moving": [1, 1, 0],
         "n_matches": [2, 3, 4],
         "n_total": [4, 4, 4],
         "fraction_matched": [0.5, 3/4, 1]})
    summary_df.attrs["warped_images"] = {
        "321_to_123": random_image_array,
        "123_to_321": random_image_array,
        "123_to_456": random_image_array,
        "456_to_123": random_image_array,
        "456_to_321": random_image_array,
        "321_to_456": random_image_array}
    summary_df.attrs["expt_id_stim_name_map"] = {123: "A", 321: "B", 456: "C"}
    summary_df.attrs["expt_id_avg_image_map"] = {
        123: random_image_array,
        321: random_image_array,
        456: random_image_array}
    return summary_df


def test_plot_container_match_fraction(nway_summary_df_fixture):
    # Smoke test to ensure that *a* plot can be generated without errors
    obt = nwdi.plot_container_match_fraction(nway_summary_df_fixture)
    assert isinstance(obt, matplotlib.figure.Figure)


def test_plot_container_warp_overlays(nway_summary_df_fixture):
    # Smoke test to ensure that *a* plot can be generated without errors
    obt = nwdi.plot_container_warp_overlays(nway_summary_df_fixture)
    assert isinstance(obt, matplotlib.figure.Figure)


def test_plot_container_warp_summary(nway_summary_df_fixture):
    # Smoke test to ensure that *a* plot can be generated without errors
    obt = nwdi.plot_container_warp_summary(nway_summary_df_fixture)
    assert isinstance(obt, matplotlib.figure.Figure)
