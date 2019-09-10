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


class NwayMatching(ArgSchemaParser):
    default_schema = NwayMatchingSchema
    default_output_schema = NwayMatchingOutputSchema

    def parse_jsons(self, input_json):
        """read the input json, populate the experiments
        with nice masks.

        Parameters
        ----------
        input_json : str
            path to json file containing nway matching input

        """
        with open(input_json, 'r') as f:
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
        '''Remove redundancy from the matching table. Redundancy include lines
           that are the same or one is the subset of another.
        '''

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
        ''' Prune matching graph to remove matching conflicts.
            This is achieved by creating a graph. Each matching line
            is a node in the graph. An edge indicates conflict
            between matchings respresented by the two nodes. The
            final result is obtained by pruning the graph based on weight
            matrix computed earlier.
        '''

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

        # define a graph with a node for each line in the table
        # and the score as an attribute
        G = nx.Graph()
        for i in range(nline):
            G.add_node(i, score=score[i])

        # if any 2 lines share an entry, make an edge between them
        edge = np.zeros((nline, nline))
        for (i0, line0), (i1, line1) in itertools.combinations(
                enumerate(matching_table_nway), 2):
            nz = (line0 != -1) & (line1 != -1)
            if np.any(line0[nz] == line1[nz]):
                G.add_edge(i0, i1)
                edge[i0, i1] = 1

        # NOTE
        # this is the original logic, rewritten using networkx graph
        # rather than the custom graph, to be more readable.
        # this logic is order-dependent!
        matching_table_nway_new = []
        nodes = list(G.nodes())
        for node in nodes:
            if node in G.nodes():
                neighbors = nx.neighbors(G, node)
                neighbor_scores = [G.nodes()[n]['score'] for n in neighbors]
                node_score = G.nodes()[node]['score']
                if np.all(node_score <= neighbor_scores):
                    # if a node has the lowest score of any of its neighbors
                    # remove the neighbors
                    [G.remove_node(n) for n in enumerate(neighbors)]
                else:
                    # otherwise, remove the node
                    G.remove_node(node)

        # any remaining nodes are lines for the pruned table
        labelval = 0
        for node in G.nodes():
            this_record = np.append(matching_table_nway[node], labelval)
            matching_table_nway_new.append(this_record.astype('int'))
            labelval += 1

        return matching_table_nway_new

    def add_remaining_cells(self, matching_table_nway):
        ''' Add remaining cells that do not find matches into the table.'''

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
                this_record = np.zeros(len(self.experiments) + 1) - 1
                this_record[i] = label_remain[i][j]
                this_record[len(self.experiments)] = cnt
                matching_table_nway_new.append(this_record.astype(int))

        return matching_table_nway_new

    def write_matching_table(self):
        ''' Write the final matching table that include all the cells in N sessions.
            Each row is one matching. It has N+1 entries. -1 means no match.
            The last entry is the unified label of the cells.'''
        filename_matching_table = os.path.join(
            self.args['output_directory'], 'matching_result.txt')
        np.savetxt(
                filename_matching_table,
                self.matching_table_nway,
                delimiter=' ',
                fmt='%d')

    def create_output_json(self):
        ''' Write the matching result into output json file. '''

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
            thisrgn = [v for v in self.matching_table_nway[i][:-1] if v != -1]

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

    def match_nway(self, para):
        '''Nway cell matching by calling pairwise
           matching and then combining the results'''

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

        return

    def run(self):
        ''' Main function of nway cell matching across multiple ophys sessions.
            The method takes three step:
            1) Image registration;
            2) Pairwise matching;
            3) Combining pairwise matching to generate Nway result.
        '''

        self.parse_jsons(self.args['input_json'])

        self.match_nway(self.args)
        logger.info("Nway matching is done!")

        self.write_matching_table()
        logger.info("Matching table is written!")

        output_json = self.create_output_json()
        self.output(output_json, indent=2)


if __name__ == "__main__":  # pragma: no cover
    nmod = NwayMatching()
    nmod.run()
