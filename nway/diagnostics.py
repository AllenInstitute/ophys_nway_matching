from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np
import itertools
from argschema import ArgSchemaParser
from nway.schemas import NwayDiagnosticSchema
import os


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
