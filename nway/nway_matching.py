# -*- coding: utf-8 -*-
'''
Main function of nway matching of cells

Copyright (c) Allen Institute for Brain Science

Usage:
--input_json <input json path>
--output_json <output json path>
--munkres_executable <munkres executable>
'''

import logging
import numpy as np
import os
import json
import itertools
import pandas as pd
import networkx as nx
from nway.pairwise_matching import PairwiseMatching
from nway.schemas import NwayMatchingSchema, NwayMatchingOutputSchema
import nway.utils as utils
from argschema import ArgSchemaParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NwayException(Exception):
    pass


def prune_matching_table_legacy(table, score):
    """eliminates match conflicts by comparing scores

    Parameters
    ----------
    table : list
        list (len = n_match_candidates) of numpy arrays (size = n_experiment)
        containing either cellIDs or -1
    score : :class:`numpy.ndarray`
        size = n_match_candidates, a float score value for each candidat

    Returns
    -------
    pruned : list
        pruned version of the input table

    """
    linenum = np.shape(table)[0]
    expnum = np.shape(table)[1]
    pruned = []

    # if two records share common elements in the
    # same column, then create an edge between them
    edge = np.zeros((linenum, linenum))
    node_exist = np.ones(linenum)

    for i in range(linenum - 1):
        for j in range(i + 1, linenum):
            for k in range(expnum):
                if (
                       (table[j][k] == table[i][k]) and
                       (table[i][k] != -1) and
                       (table[j][k] != -1)):
                    edge[i, j] = 1
                    break

    labelval = 0
    for i in range(linenum):
        if node_exist[i] == 1:
            idx = np.argwhere(edge[i, :] == 1)
            len_idx = len(idx)
            if len_idx == 0:  # no conflict with other matching
                labelval = labelval + 1
                pruned.append(table[i])

            # score is the smallest, equal may lead
            # to conflict matching added to the list?
            elif score[i] <= np.min(score[idx]):
                labelval = labelval + 1
                pruned.append(table[i])

                # remove the nodes connected to it
                # as they have worse scores
                edge[i, idx] = 0
                for k in range(len_idx):
                    edge[idx[k], :] = 0
                node_exist[idx] = 0

            else:  # prune the edge
                edge[i, idx] = 0

    return pruned


def subgraphs(G):
    """returns connected component subgraphs. Replaces deprecated
    nx.connected_component_subgraphs()

    Parameters
    ----------
    G : :class:`networkx.Graph`

    Returns
    -------
    subgs : tuple
        tuple of connected subgraphs of G

    """
    subgs = (G.subgraph(c).copy() for c in nx.connected_components(G))
    return subgs


def greduce(glist, method='keepmin'):
    """recursively reduces a graph according to node attribute 'score'

    Parameters
    ----------
    glist : tuple
        tuple of subgraphs
    method : str
        'keepmin' reduce graph by finding the minimum score of a subraph
            and eliminating its neighbors
        'popmax' reduce graph by finding the maximum score and eliminating it

    Returns
    -------
    results : list
        list of nodes that survive the pruning

    """
    results = []
    for g in glist:
        gnodes = list(g.nodes.keys())
        if len(gnodes) == 1:
            results += [gnodes[0]]
        else:
            if method == 'popmax':
                maxind = np.argmax([i['score'] for i in g.nodes.values()])
                g.remove_node(gnodes[maxind])
            elif method == 'keepmin':
                minind = np.argmin([i['score'] for i in g.nodes.values()])
                neighbors = list(g.neighbors(gnodes[minind]))
                for n in neighbors:
                    g.remove_node(n)
            results += greduce(subgraphs(g), method=method)

    return results


def prune_matching_table(table, score, method='keepmin'):
    """eliminates match conflicts by comparing scores

    Parameters
    ----------
    table : list
        list (len = n_match_candidates) of numpy arrays (size = n_experiment)
        containing either cellIDs or -1
    score : :class:`numpy.ndarray`
        size = n_match_candidates, a float score value for each candidat
    method : str
        'keepmin' reduce graph by finding the minimum score of a subraph
            and eliminating its neighbors
        'popmax' reduce graph by finding the maximum score and eliminating it

    Returns
    -------
    pruned : list
        pruned version of the input table

    """
    G = nx.Graph()
    for i in range(len(table)):
        G.add_node(i, score=score[i])

    # if any 2 lines share an entry, make an edge between them
    for (i0, line0), (i1, line1) in itertools.combinations(
            enumerate(table), 2):
        nz = (line0 != -1) & (line1 != -1)
        if np.any(line0[nz] == line1[nz]):
            G.add_edge(i0, i1)

    inds = greduce(subgraphs(G), method=method)
    pruned = [table[i] for i in inds]

    return pruned


class NwayMatching(ArgSchemaParser):
    default_schema = NwayMatchingSchema
    default_output_schema = NwayMatchingOutputSchema

    def parse_data_json(self, input_data_json):
        """read the input json, populate the experiments
        with nice masks.

        Parameters
        ----------
        input_json : str
            path to json file containing nway matching input

        """
        with open(input_data_json, 'r') as f:
            data = json.load(f)

        if self.args['output_directory'] is None:
            self.args['output_directory'] = str(data['output_directory'])

        self.experiments = []
        for exp in data['experiment_containers']['ophys_experiments']:
            self.experiments.append(
                    utils.create_nice_mask(
                        exp,
                        self.args['output_directory'],
                        legacy_label_order=self.args['legacy_label_order']))

        if len(self.experiments) < 2:
            raise NwayException("Need at least 2 experiements from input")

    def gen_nway_table_with_redundancy(self):
        """Combine all pairwise matches into one table
        regardless of conflicts.

        """
        matching_frame = pd.DataFrame(
            columns=[e['id'] for e in self.experiments])

        for pair in self.pair_matches:
            with open(pair.args['output_json'], 'r') as f:
                pairj = json.load(f)
            pairframe = pd.DataFrame(
                    [[i['fixed'], i['moving']] for i in pairj['matches']],
                    columns=[
                        pairj['fixed_experiment'],
                        pairj['moving_experiment']])
            matching_frame = matching_frame.append(pairframe)

        return matching_frame

    @staticmethod
    def remove_nway_table_redundancy(frame):
        """Remove any matching lines or subsets in a table

        Parameters
        ----------
        frame : :class:`pandas.DataFrame` or :class:`numpy.ndarray`
            table of cellIDs or zeros

        Returns
        -------
        table_new : list
            list of lines without redundancy or subsets

        """

        table = np.array(frame).astype('float')
        table[np.isnan(table)] = -1
        table = table.astype('int')

        linenum, expnum = np.shape(table)
        stoptag = 0

        while stoptag == 0:
            table_new = [table[0]]

            for i, oline in enumerate(table[1:], 1):
                for j, nline in enumerate(table_new):
                    mergetag = 1
                    omiss = oline == -1
                    nmiss = nline == -1

                    # matches not in any of the same experiments
                    if np.all(omiss | nmiss):
                        mergetag = 0

                    # matches in the same experiment(s), but no match
                    if np.any(
                            (nline != oline) &
                            np.invert(omiss) &
                            np.invert(nmiss)):
                        mergetag = 0

                    if mergetag == 1:
                        for k in range(expnum):
                            if oline[k] != -1:
                                nline[k] = oline[k]
                        break
                        # break for loop j
                        # only merge to one previous record

                # old line unrelated to any new lines, add it
                if mergetag == 0:
                    table_new.append(oline)

            linenum_new = np.shape(table_new)[0]

            if linenum == linenum_new:
                stoptag = 1
            else:
                table = table_new
                linenum = linenum_new

        return table_new

    def prune_matching_graph(self, matching_table_nway):
        """remove matching conflicts by scoring match candidates
        and pruning according to that score

        Parameters
        ----------
        matching_table_nway : list
            list of match arrays

        Returns
        -------
        matching_table_nway_new : list
            pruned list of match arrays

        """

        nline = np.shape(matching_table_nway)[0]

        exids = [exp['id'] for exp in self.experiments]

        # NOTE
        # assign a score to each line in the matching table
        # the score is the sum of costs for each pair in the
        # line. The costs are generally in [0, 2] when the
        # pair was within max_distance and 1000 when the pair
        # was not.
        # the scores are used to choose between conflicting lines
        # for n experiments, there are np = (n)(n - 1) / 2 pairs
        # example 1:
        # line1 = [a, b, -1, -1], line2 = [a, c, -1, -1]
        # score1 = cost(a, b) + 5000, score2 = cost(a, c) + 5000
        # if cost(a, b) < cost(a, c), line1 is preferred
        # example 2:
        # line1 = [a, b, c, -1], line2 = [a, c, -1, -1]
        # score1 = cost(a, b) + cost(b, c) + cost(a, c) + 3000
        # score2 = cost(a, c) + 5000
        # score1 in range [3000, 3006], score2 in range [5000, 5002]
        # line1 is preferred

        score = np.zeros(nline)
        for pair in self.pair_matches:
            with open(pair.args['output_json'], 'r') as f:
                j = json.load(f)
            # all possible candidates within max_distance
            candidates = {
                    (m['fixed'], m['moving']): m['cost']
                    for k in ['matches', 'rejected']
                    for m in j[k]}
            prow = exids.index(pair.args['fixed']['id'])
            pcol = exids.index(pair.args['moving']['id'])
            for i in range(nline):
                col = matching_table_nway[i][pcol]
                row = matching_table_nway[i][prow]
                if self.args['legacy_pruning_index_error']:
                    # indexing mistake in original code
                    if col == -1:
                        col = pair.cost_matrix.columns[-2]
                    if row == -1:
                        row = pair.cost_matrix.index[-2]
                    pscore = pair.cost_matrix[col][row]
                else:
                    if (row, col) in candidates:
                        pscore = candidates[(row, col)]
                    else:
                        pscore = 1000
                score[i] += pscore

        if self.args['legacy_pruning_order_dependence']:
            matching_table_nway_new = prune_matching_table_legacy(
                    matching_table_nway, score)
        else:
            matching_table_nway_new = prune_matching_table(
                    matching_table_nway, score, self.args['pruning_method'])

        return matching_table_nway_new

    def add_remaining_cells(self, matching_table_nway):
        """add any remaining cells that had no match

        Parameters
        ----------
        matching_table_nway : list
            list of match arrays

        Returns
        -------
        matching_table_nway_new : list
            with added unmatched cells

        """

        linenum = np.shape(matching_table_nway)[0]
        cnt = linenum
        matching_table_nway_new = matching_table_nway
        label_remain = []

        for j in range(len(self.experiments)):
            labels = [v['id'] for v in self.experiments[j]['cell_rois']]
            for i in range(linenum):
                labels = np.setdiff1d(labels, [matching_table_nway[i][j]])
            label_remain.append(labels)

        for i in range(len(self.experiments)):
            num = len(label_remain[i])
            for j in range(num):
                cnt = cnt + 1
                this_record = np.zeros(len(self.experiments)) - 1
                this_record[i] = label_remain[i][j]
                matching_table_nway_new.append(this_record.astype(int))

        return matching_table_nway_new

    def create_output_dict(self):
        """write the nway matching results to a dict

        Returns
        -------
        matchingdata: dict
           "nway_matches": list of sets of cellID matches
           "pairwise_results": list of results from the pairwise matching
        """
        cellnum = np.shape(self.matching_table_nway)[0]
        matchingdata = dict()
        matchingdata["nway_matches"] = []
        prob = np.zeros(len(self.experiments))

        for i in range(cellnum):
            matching_exp_num = \
                    len(self.experiments) - 1 - \
                    np.count_nonzero(self.matching_table_nway[i] == -1)
            prob[matching_exp_num] = prob[matching_exp_num] + 1
            logger.debug(self.matching_table_nway[i])

        for i in range(cellnum):
            thisrgn = [v for v in self.matching_table_nway[i] if v != -1]

            if len(thisrgn) > 0:
                matchingdata["nway_matches"].append(thisrgn)

        matchingdata["pairwise_results"] = []
        for k in self.pair_matches:
            with open(k.args['output_json'], 'r') as f:
                j = json.load(f)
            matchingdata['pairwise_results'].append(j)
            if not self.args['save_pairwise_results']:
                os.remove(k.args['output_json'])

        return matchingdata

    def run(self):
        """Nway cell matching by calling pairwise
           matching and then combining the results
        """

        self.parse_data_json(self.args['input_data_json'])

        # pair-wise matching
        self.pair_matches = []
        for fixed, moving in itertools.combinations(self.experiments, 2):
            pair_args = dict(self.args)
            pair_args["fixed"] = fixed
            pair_args["moving"] = moving
            pair_args["output_json"] = os.path.join(
                    self.args["output_directory"],
                    "{}_to_{}_output.json".format(moving['id'], fixed['id']))
            self.pair_matches.append(
                    PairwiseMatching(input_data=pair_args, args=[]))
            self.pair_matches[-1].run()

        # generate N-way matching table
        matching_frame = self.gen_nway_table_with_redundancy()

        matching_table_nway_tmp = self.remove_nway_table_redundancy(
                matching_frame)

        matching_table_nway_tmp = self.prune_matching_graph(
                matching_table_nway_tmp)

        self.matching_table_nway = self.add_remaining_cells(
                matching_table_nway_tmp)

        logger.info("Nway matching is done!")

        output_dict = self.create_output_dict()
        self.output(output_dict, indent=2)
        logger.info("wrote {}".format(self.args['output_json']))


if __name__ == "__main__":  # pragma: no cover
    nmod = NwayMatching()
    nmod.run()
