import pytest
import matplotlib
import os
import PyPDF2
import shutil
import json
import nway.diagnostics as nwdi
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

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
