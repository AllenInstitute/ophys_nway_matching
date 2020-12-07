import itertools
import json
import os
import time
from pathlib import Path
from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
from argschema import ArgSchemaParser
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from skimage.metrics import structural_similarity

import nway.image_processing_utils as imutils
from nway.schemas import NwayDiagnosticSchema, NwayMatchSummarySchema


def cell_experiment_dict(nway_output_path):
    """lookup dict for experiment by cell ID

    Parameters
    ----------
    nway_output_path : str
        path to output json from NwayMatching

    Returns
    -------
    lookup : dict
        dictionary with "cellID": "experimentID"

    """
    with open(nway_output_path, 'r') as f:
        j = json.load(f)
    lookup = {}
    for pw in j['pairwise_results']:
        print(list(pw.keys()))
        for k in ['rejected', 'matches']:
            for pair in pw[k]:
                lookup[pair['fixed']] = pw['fixed_experiment']
                lookup[pair['moving']] = pw['moving_experiment']
        for cellid in pw['unmatched']['fixed']:
            lookup[cellid] = pw['fixed_experiment']
        for cellid in pw['unmatched']['moving']:
            lookup[cellid] = pw['moving_experiment']
    return lookup


def pairwise_transforms(
        nway_output_path, fig=None, subplot_spec=None, fontsize=6):
    """summarize transform parameters and optionally plot

    Parameters
    ----------
    nway_output_path : str
        path to output json from NwayMatching
    fig : :class:`matplotlib.figure.Figure`
        destination figure for plots. No plotting if None
    subplot_spec : :class:`matplotlib.gridspec.SubplotSpec`
        destination SubplotSpec for plots. If None, subplots
        consumer entire figure
    fontsize : int
        fontsize for text in plot

    Returns
    -------
    results : dict
        summarized pairwise transform parameters and experiment ids

    """
    # read the results into lists
    with open(nway_output_path, 'r') as f:
        j = json.load(f)
    pairs = j['pairwise_results']
    props = list(pairs[0]['transform']['properties'].keys())
    results = {}
    for k in props:
        results[k] = [pair['transform']['properties'][k] for pair in pairs]
    results['ids'] = [
            "%d-\n%d" % (
                pair['moving_experiment'],
                pair['fixed_experiment']) for pair in pairs]

    if fig is not None:
        # plot, if a figure is provided
        if subplot_spec is None:
            # these subplots fill entire figure
            outer_plot_grid = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
            subplot_spec = outer_plot_grid[0]

        inner_plot_grid = gridspec.GridSpecFromSubplotSpec(
                len(props), 1, subplot_spec=subplot_spec,
                wspace=0.1, hspace=0.1)

        x = np.arange(len(results['ids']))
        labels = ['x', 'y']
        for ip, prop in enumerate(props):
            ax = fig.add_subplot(inner_plot_grid[ip])
            ares = np.array(results[prop])
            if len(ares.shape) == 1:
                ares = ares.reshape(-1, 1)
            ares = ares.transpose()
            for iv, v in enumerate(ares):
                ax.plot(x, v, '-o', label=labels[iv])
                ax.set_ylabel(prop, fontsize=fontsize)
            if ares.shape[0] == 2:
                ax.legend(ncol=2, fontsize=fontsize, loc='best', frameon=False)
            ax.set_xticks(x)
            ax.set_xticklabels([])
            if ip == 0:
                ax.set_title(
                        'pairwise transform properties', fontsize=fontsize)
        ax.set_xticklabels(results['ids'], rotation=45, fontsize=fontsize)
        ax.set_xlabel('pairwise experiment IDs', fontsize=fontsize)

    return results


def some_grid(n):
    """specify a roughly square grid with n elements

    Parameters
    ----------
    n : int
        number of elements

    Returns
    -------
    (nrow, ncol) : tuple
        shape for new grid

    """
    nrow = int(np.floor(np.sqrt(n)))
    # NOTE float cast for python 2.7
    ncol = int(np.ceil(float(n) / nrow))
    return (nrow, ncol)


def pairwise_matches(
        nway_output_path, fig=None, subplot_spec=None, fontsize=6):
    """summarize pairwise matches and reject costs

    Parameters
    ----------
    nway_output_path : str
        path to output json from NwayMatching
    fig : :class:`matplotlib.figure.Figure`
        destination figure for plots. No plotting if None
    subplot_spec : :class:`matplotlib.gridspec.SubplotSpec`
        destination SubplotSpec for plots. If None, subplots
        consumer entire figure
    fontsize : int
        fontsize for text in plot

    Returns
    -------
    costs : dict
        costs[pair id] = {
            'matches': list of costs,
            'rejected': list of costs}
    allcosts = dict
        'matches' : concat of all pair matches,
        'rejected' : concate of all pair rejected

    """
    # read the results into lists
    with open(nway_output_path, 'r') as f:
        j = json.load(f)
    pairs = j['pairwise_results']

    costs = {}
    subkeys = ['matches', 'rejected']
    for pair in pairs:
        k = "%d-%d" % (pair['moving_experiment'], pair['fixed_experiment'])
        costs[k] = {}
        for subk in subkeys:
            costs[k][subk] = [m['cost'] for m in pair[subk]]
    allcosts = {}
    for subk in subkeys:
        allcosts[subk] = np.concatenate([c[subk] for c in costs.values()])

    if fig is not None:
        # plot, if a figure is provided
        if subplot_spec is None:
            # these subplots fill entire figure
            outer_plot_grid = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
            subplot_spec = outer_plot_grid[0]

        inner_plot_grid = gridspec.GridSpecFromSubplotSpec(
                *some_grid(len(costs)), subplot_spec=subplot_spec,
                wspace=0.1, hspace=0.1)

        bins = np.arange(0, 2.1, 0.1)

        for ic, ck in enumerate(costs):
            ax = fig.add_subplot(inner_plot_grid[ic])
            ax.hist(
                    allcosts['matches'],
                    histtype='step',
                    color='k',
                    bins=bins,
                    label='all matches')
            ax.hist(
                    allcosts['rejected'],
                    histtype='step',
                    color='r',
                    bins=bins,
                    label='all rejected')
            ax.hist(
                    costs[ck]['matches'],
                    color='k',
                    alpha=0.5,
                    bins=bins,
                    label='pair matches')
            ax.hist(
                    costs[ck]['rejected'],
                    color='r',
                    alpha=0.5,
                    bins=bins,
                    label='pair rejected')
            ax.legend(loc=2, fontsize=fontsize, title=ck, frameon=False)
            ax.get_legend().get_title().set_fontsize(fontsize)
            ax.set_yscale('log')
            ax.set_xlabel('cost', fontsize=6)
            ax.set_ylabel('match count', fontsize=6)
            rc = inner_plot_grid[ic].get_rows_columns()
            if rc[3] != (rc[0] - 1):
                ax.set_xticks([])
            if rc[5] != 0:
                ax.set_yticks([])

    return costs, allcosts


def nway_matches(
        nway_output_path, fig=None, fontsize=12):
    """summarize pairwise matches and reject costs

    Parameters
    ----------
    nway_output_path : str
        path to output json from NwayMatching
    fig : :class:`matplotlib.figure.Figure`
        destination figure for plots. No plotting if None
    fontsize : int
        fontsize for text in plot

    Returns
    -------
    allnw : dict
        keys are sorted tuples of match sets
        values are :
            n : number of pairwise matches (< max_distance)
            avecost : average pairwise cost

    """

    _, allcosts = pairwise_matches(nway_output_path)

    # read the results into lists
    with open(nway_output_path, 'r') as f:
        j = json.load(f)
    pairs = j['pairwise_results']
    nway = j['nway_matches']

    # make a set of tuples of all pairwise matches and rejected
    # (anything within max_distance)
    # keep the tuples ordered so we can search easily
    allpw = {}
    for pair in pairs:
        for subk in ['matches', 'rejected']:
            pw = {tuple(np.sort([m['moving'], m['fixed']])): m['cost']
                  for m in pair[subk]}
            allpw.update(pw)

    # go through all the nway matches and get the n and mean for each
    # entry
    allnw = {}
    for match in nway:
        k = tuple(match)
        n = 0
        avcost = 0
        for candidate in itertools.combinations(match, 2):
            tsort = tuple(np.sort(candidate))
            if tsort in allpw:
                n += 1
                avcost += allpw[tsort]
        if n != 0:
            allnw[k] = {
                    'n': n,
                    'avecost': avcost / n}

    if fig is not None:
        spec = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)

        ax0 = fig.add_subplot(spec[1:, 0:4])
        ax_histy = fig.add_subplot(spec[1:5, 4:])
        ax_histx = fig.add_subplot(spec[0, 0:4])

        ns = np.array([i['n'] for i in allnw.values()])
        x = np.arange(ns.min(), ns.max() + 1)
        bins = np.arange(0.0, 2.1, 0.1)
        y = []
        cbinned = []
        for ix in x:
            iy = [i['avecost'] for i in allnw.values() if i['n'] == ix]
            y.append(iy)
            h, _ = np.histogram(iy, bins=bins)
            cbinned.append(h)
        cbinned = np.array(cbinned)

        ax0.imshow(
                np.flipud(cbinned),
                extent=[
                    bins.min(),
                    bins.max(),
                    ns.min() - 0.5,
                    ns.max() + 0.5],
                cmap='gray_r',
                aspect='auto')

        xbins = [bins[i:i+2].mean() for i in range(bins.size - 1)]
        ax_histx.bar(
                xbins,
                cbinned.sum(axis=0),
                color='k',
                alpha=0.5,
                width=0.1,
                edgecolor='k')
        ax_histy.barh(
                x,
                cbinned.sum(axis=1),
                color='k',
                alpha=0.5,
                height=1.0,
                edgecolor='k')

        ax0.set_xlabel('cost', fontsize=fontsize)
        ax0.set_ylabel('matches per set', fontsize=fontsize)
        ax_histx.set_xticks([])
        ax_histy.set_yticks([])
        ax_histy.set_xlabel('count', fontsize=fontsize)
        ax_histx.set_ylabel('count', fontsize=fontsize)
        ax_histx.set_title('nway match results', fontsize=fontsize)
        ax_histx.set_xlim(ax0.get_xlim())
        ax_histy.set_ylim(ax0.get_ylim())

    return allnw


def plot_all(nway_output_path, fname=None):
    fs = (12, 8)

    fig1 = plt.figure(clear=True, figsize=fs)
    pairwise_transforms(nway_output_path, fig=fig1)

    fig2 = plt.figure(clear=True, figsize=fs)
    pairwise_matches(nway_output_path, fig=fig2)

    fig3 = plt.figure(clear=True, figsize=fs)
    nway_matches(nway_output_path, fig=fig3)

    if fname is not None:
        p = PdfPages(fname)
        p.savefig(fig1)
        p.savefig(fig2)
        p.savefig(fig3)
        p.close()


def create_nway_input_maps(nway_input: dict) -> Tuple[dict, dict]:
    """Create mappings between experiment id, experiment stimulus name,
    and the average intensity projection image arrays.

    Parameters
    ----------
    nway_input : dict
        The 'input.json' in dictionary form passed to the nway matching
        module.

    Returns
    -------
    Tuple[dict, dict]
        A tuple of two mappings. The first mapping relates experiment ids to
        experiment stimulus names. The second mapping relates experiment ids
        to the average intensity projection image array for the experiment.
    """
    ophys_expts = nway_input['experiment_containers']['ophys_experiments']

    expt_id_stim_name_map = dict()
    expt_id_avg_image_map = dict()
    for expt in ophys_expts:
        stimulus_name = expt.get('stimulus_name')
        if stimulus_name is None:
            stimulus_name = 'Unknown Stimulus'
        expt_id_stim_name_map[expt['id']] = stimulus_name

        avg_image_key = 'ophys_average_intensity_projection_image'
        with PIL.Image.open(expt[avg_image_key]) as im:
            expt_avg_image = np.array(im)
        expt_id_avg_image_map[expt['id']] = expt_avg_image

    expt_id_stim_name_map = {k: v for k, v
                             in sorted(expt_id_stim_name_map.items(),
                                       key=lambda x: str(x[1]))}

    return (expt_id_stim_name_map, expt_id_avg_image_map)


def create_nway_summary_df(expt_id_stim_name_map: dict,
                           expt_id_avg_image_map: dict,
                           nway_output: dict) -> pd.DataFrame:
    """Create an nway matching summary dataframe necessary for plotting
    match fractions and assessing average image registrations.

    Parameters
    ----------
    expt_id_stim_name_map : dict
        A mapping that relates experiment ids to experiment stimulus names.
        Produced by 'create_nway_input_maps'.
    expt_id_avg_img_map : dict
        A mapping that relates experiment ids to the average intensity
        projection image array for the experiment. Produced by
        'create_nway_input_maps'.
    nway_output : dict
        The 'output.json' in dictionary form produced by the nway matching
        module.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns:

        fixed_expt (int): The experiment id of the alignment target
        fixed_expt_stim_name (str): The stimulus name for the fixed expt
        moving_expt (int): The experiment id of the image to align
        moving_expt_stim_name (str): The stimulus name for the moving expt
        n_unmatched_fixed (int): Number of ROIs from the fixed experiment which
            could not be matched to a moving experiment's ROI
        n_unmatched_moving (int): Number of ROIs from the moving experiment
            which could not be matched to a fixed experiment's ROI
        n_matches (int): Number of ROIs that were matched between fixed and
            moving experiments
        n_total (int): The total number of ROIs to match
        fraction_matched (float): n_matches divided by n_total

        This DataFrame additionally contains the following attributes keys:

        warped_images: A mapping that relates a pairwise match to a warped
            registration image
        expt_id_stim_name_map: A mapping that relates an experiment's id with a
            stimulus name (describing the type of experiment)
        expt_id_avg_image_map: A mapping that relates an experiment's id with
            the experiment's average intensity projection image array
    """
    pairwise_results = nway_output['pairwise_results']

    warped_avg_image_maps = dict()
    df_list = []
    for pair in pairwise_results:
        # Assemble match statistics
        fixed_expt = pair['fixed_experiment']
        fixed_expt_stim_name = expt_id_stim_name_map[fixed_expt]

        moving_expt = pair['moving_experiment']
        moving_expt_stim_name = expt_id_stim_name_map[moving_expt]

        n_unmatched_fixed = len(pair['unmatched']['fixed'])
        n_unmatched_moving = len(pair['unmatched']['moving'])
        n_matches = len(pair['matches'])
        n_total = n_matches + n_unmatched_fixed + n_unmatched_moving
        fraction_matched = n_matches / float(n_total)
        df_list.append([fixed_expt, fixed_expt_stim_name, moving_expt,
                        moving_expt_stim_name, n_unmatched_fixed,
                        n_unmatched_moving, n_matches, n_total,
                        fraction_matched])

        # Recreate warped images
        transform_matrix = np.array(pair['transform']['matrix'])
        transform_type = pair['transform']['transform_type']
        fixed_img = expt_id_avg_image_map[fixed_expt]
        moving_img = expt_id_avg_image_map[moving_expt]

        moving_warped = imutils.warp_image(moving_img, transform_matrix,
                                           transform_type, fixed_img.shape)
        warped_img_key_1 = f"{moving_expt}_to_{fixed_expt}"
        warped_avg_image_maps[warped_img_key_1] = moving_warped

        inv_transform = np.linalg.inv(transform_matrix)
        fixed_warped = imutils.warp_image(fixed_img, inv_transform,
                                          transform_type, moving_img.shape)
        warped_img_key_2 = f"{fixed_expt}_to_{moving_expt}"
        warped_avg_image_maps[warped_img_key_2] = fixed_warped

    columns = ['fixed_expt', 'fixed_expt_stim_name', 'moving_expt',
               'moving_expt_stim_name', 'n_unmatched_fixed',
               'n_unmatched_moving', 'n_matches', 'n_total',
               'fraction_matched']
    nway_summary_df = pd.DataFrame(df_list, columns=columns)
    nway_summary_df = nway_summary_df.sort_values(by=['fixed_expt_stim_name',
                                                      'moving_expt_stim_name'],
                                                  ignore_index=True)
    nway_summary_df.attrs['warped_images'] = warped_avg_image_maps
    nway_summary_df.attrs['expt_id_stim_name_map'] = expt_id_stim_name_map
    nway_summary_df.attrs['expt_id_avg_image_map'] = expt_id_avg_image_map

    return nway_summary_df


def plot_container_match_fraction(nway_summary_df: pd.DataFrame) -> Figure:
    """Given an nway summary DataFrame, produce a plot summarizing ROI
    match fractions.
    """
    expt_id_stim_name_map = nway_summary_df.attrs['expt_id_stim_name_map']
    expt_ids = expt_id_stim_name_map.keys()

    match_frac_mtx = pd.DataFrame(index=expt_ids,
                                  columns=expt_ids,
                                  dtype=float)
    for _, row in nway_summary_df.iterrows():
        match_fraction = row['fraction_matched']
        match_frac_mtx[row['fixed_expt']][row['moving_expt']] = match_fraction
        match_frac_mtx[row['moving_expt']][row['fixed_expt']] = match_fraction
    np.fill_diagonal(match_frac_mtx.values, 1.0)

    fig, ax = plt.subplots(figsize=(12, 12))
    mat_ax = ax.matshow(match_frac_mtx, vmin=0.0, vmax=1.0, cmap='magma')

    for (i, j), data in np.ndenumerate(match_frac_mtx.values):
        if i == j:
            ax.text(j, i, f'{data:.3f}', ha='center', va='center',
                    fontsize=18)
        else:
            ax.text(j, i, f'{data:.3f}', ha='center', va='center',
                    fontsize=18, color='white')

    ax.xaxis.set_ticks_position('bottom')
    xy_labels = [f"{stim_name} (Expt: {expt_id})"
                 for expt_id, stim_name in expt_id_stim_name_map.items()]
    ax.set_xticks(list(range(len(expt_ids))))
    ax.set_xticklabels(xy_labels, fontsize=18, rotation=45, ha='right')
    ax.set_yticks(list(range(len(expt_ids))))
    ax.set_yticklabels(xy_labels, fontsize=18)

    plt.title("Fraction matched ROIs across sessions", fontsize=24, pad=20)
    fig.colorbar(mat_ax)
    return fig


def plot_container_warp_overlays(nway_summary_df: pd.DataFrame) -> Figure:
    """Given an nway summary DataFrame, produce a plot that shows the
    overlap of experiment average intensity projection images after
    registration.
    """
    expt_id_stim_name_map = nway_summary_df.attrs['expt_id_stim_name_map']
    expt_ids = expt_id_stim_name_map.keys()

    expt_id_avg_image_map = nway_summary_df.attrs['expt_id_avg_image_map']
    warped_images_map = nway_summary_df.attrs['warped_images']

    panel_len = len(expt_ids) + 1

    fig, axes = plt.subplots(nrows=panel_len,
                             ncols=panel_len,
                             figsize=(25, 25))
    # Turn off all axes in subplots
    for ax in axes.ravel():
        ax.set_axis_off()

    # plot unwarped base images
    for idx, _id in enumerate(expt_ids, start=1):
        session_type = '_'.join(expt_id_stim_name_map[_id].split('_')[:2])

        axes[0][idx].imshow(expt_id_avg_image_map[_id], cmap='gray')
        axes[0][idx].set_title(f"Expt: {_id}\n{session_type}", fontsize=18)
        axes[idx][0].imshow(expt_id_avg_image_map[_id], cmap='gray')
        axes[idx][0].set_title(f"Expt: {_id}\n{session_type}",
                               x=-0.5, y=0.4, fontsize=18)

    # plot warped 'moving' expt image on 'fixed' expt image
    for row, expt_1 in enumerate(expt_ids, start=1):
        for col, expt_2 in enumerate(expt_ids, start=1):
            if expt_1 == expt_2:
                continue
            else:
                expt_1_avg_img = expt_id_avg_image_map[expt_1]
                norm_expt_1_avg_img = (
                    expt_1_avg_img / float(np.amax(expt_1_avg_img)))

                warp_key = f"{expt_2}_to_{expt_1}"
                warped_avg_img = warped_images_map[warp_key]

                norm_warped_avg_img = (
                    warped_avg_img / float(np.amax(warped_avg_img)))

                img_shape = norm_expt_1_avg_img.shape
                combined_img = np.zeros((img_shape[0], img_shape[1], 3))
                combined_img[:, :, 0] = norm_expt_1_avg_img
                combined_img[:, :, 1] = norm_warped_avg_img

                axes[row][col].imshow(combined_img)
                ssim = structural_similarity(norm_expt_1_avg_img,
                                             norm_warped_avg_img,
                                             gaussian_weights=True)
                axes[row][col].set_title(f"SSIM: {ssim:.3f}", fontsize=16)

    fig.tight_layout()
    return fig


def plot_container_warp_summary(nway_summary_df: pd.DataFrame) -> Figure:
    """Given an nway summary DataFrame, produce a plot that shows in
    greater detail the quality of the registration between experiment
    average intensity projection images.
    """
    expt_id_avg_image_map = nway_summary_df.attrs['expt_id_avg_image_map']
    warped_images_map = nway_summary_df.attrs['warped_images']

    num_ax_cols = len(nway_summary_df.index)

    fig, axes = plt.subplots(nrows=4,
                             ncols=num_ax_cols,
                             figsize=(30, 10))
    # Turn off all axes in subplots
    for ax in axes.ravel():
        ax.set_axis_off()

    for idx, row in nway_summary_df.iterrows():
        fixed_expt = row['fixed_expt']
        moving_expt = row['moving_expt']

        moving_image = expt_id_avg_image_map[moving_expt]
        fixed_image = expt_id_avg_image_map[fixed_expt]
        warp_key = f"{moving_expt}_to_{fixed_expt}"
        warped_image = warped_images_map[warp_key]

        norm_fixed_image = fixed_image / float(np.amax(fixed_image))
        norm_warped_image = warped_image / float(np.amax(warped_image))
        img_shape = norm_fixed_image.shape
        combined_img = np.zeros((img_shape[0], img_shape[1], 3))
        combined_img[:, :, 0] = norm_fixed_image
        combined_img[:, :, 1] = norm_warped_image

        ssim = structural_similarity(fixed_image, warped_image,
                                     gaussian_weights=True)

        moving_stimulus_name = row['moving_expt_stim_name']
        moving_session_type = '_'.join(moving_stimulus_name.split('_')[:2])
        fixed_stimulus_name = row['fixed_expt_stim_name']
        fixed_session_type = '_'.join(fixed_stimulus_name.split('_')[:2])

        axes[0][idx].imshow(moving_image, cmap='gray')
        axes[0][idx].set_title(f"Moving\n{moving_expt}\n{moving_session_type}")

        axes[1][idx].imshow(fixed_image, cmap='gray')
        axes[1][idx].set_title(f"Fixed\n{fixed_expt}\n{fixed_session_type}")

        axes[2][idx].imshow(warped_image, cmap='gray')
        axes[2][idx].set_title(f"Registered\n{moving_expt}")

        axes[3][idx].imshow(combined_img)
        axes[3][idx].set_title(f"SSIM\n{ssim:.3f}", fontsize=16)

    fig.tight_layout()
    return fig


class NwaySummary(ArgSchemaParser):
    default_schema = NwayMatchSummarySchema

    def run(self) -> dict:
        input_maps = create_nway_input_maps(self.args['nway_input'])
        expt_id_stim_name_map, expt_id_avg_image_map = input_maps

        summary_df = create_nway_summary_df(expt_id_stim_name_map,
                                            expt_id_avg_image_map,
                                            self.args['nway_output'])

        fig1 = plot_container_match_fraction(summary_df)
        fig2 = plot_container_warp_overlays(summary_df)
        fig3 = plot_container_warp_summary(summary_df)

        save_dir = Path(self.args['output_directory'])
        timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        fig1_save_path = save_dir / f"nway_match_fraction_plot_{timestamp}.png"
        fig2_save_path = save_dir / f"nway_warp_overlay_plot_{timestamp}.png"
        fig3_save_path = save_dir / f"nway_warp_summary_plot_{timestamp}.png"

        fig1.savefig(fig1_save_path, dpi=300, bbox_inches="tight")
        fig2.savefig(fig2_save_path, dpi=300, bbox_inches="tight")
        fig3.savefig(fig3_save_path, dpi=300, bbox_inches="tight")

        return {
            "nway_match_fraction_plot": str(fig1_save_path),
            "nway_warp_overlay_plot": str(fig2_save_path),
            "nway_warp_summary_plot": str(fig3_save_path)
        }


class NwayDiagnostics(ArgSchemaParser):
    default_schema = NwayDiagnosticSchema

    def run(self):
        if self.args['use_input_dir']:
            self.args['output_pdf'] = os.path.join(
                    os.path.dirname(self.args['input_json']),
                    os.path.basename(self.args['output_pdf']))

        plot_all(self.args['input_json'], fname=self.args['output_pdf'])


if __name__ == "__main__":  # pragma: no cover
    nd = NwayDiagnostics()
    nd.run()
