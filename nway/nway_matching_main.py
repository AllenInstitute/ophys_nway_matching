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
import json
import numpy as np
import SimpleITK as sitk
import os
import re
from nway.pairwise_matching import PairwiseMatching, read_tiff_3d
from nway.schemas import NwayMatchingSchema
from argschema import ArgSchemaParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NwayMatching(ArgSchemaParser):
    default_schema = NwayMatchingSchema
    ''' Class for matching cells across arbitrary number of ophys sessions.

        Necessary files are obtained by parsing input json. Final Nway matching
        result is obtained by combining pairwise matching.
    '''

    def parse_jsons_with_reference_keyword(self, input_json):
        ''' Parse input json file to genearte the necessary input files
            for nway cell matching. This function deals with the old
            input json file format, with the 'reference_experiment' keyword.
        '''

        json_data = open(input_json).read()
        data = json.loads(json_data)

        self.args['output_directory'] = str(data['output_directory'])
        self.expnum = len(data['experiment_containers']['ophys_experiments'])

        self.filename_intensity = ["" for x in range(self.expnum)]
        self.filename_segmask = ["" for x in range(self.expnum)]
        self.filename_exp_prefix = ["" for x in range(self.expnum)]

        if self.expnum < 2:
            raise RuntimeError(
                    "There should be at least two "
                    "experiments! Check input json.")

        count = 1
        for i in range(self.expnum):

            this_exp = data['experiment_containers']['ophys_experiments'][i]

            reference_experiment = this_exp['reference_experiment']
            if str(reference_experiment).upper() == "TRUE":
                # the reference is #0
                # str() convert unicode string to regular string
                self.filename_intensity[0] = \
                    str(this_exp['ophys_average_intensity_projection_image'])
                self.filename_segmask[0] = str(this_exp['max_int_mask_image'])
                ind = self.filename_intensity[0].find('ophys_experiment_')
                self.filename_exp_prefix[0] = \
                    self.filename_intensity[0][ind:ind + 26]
            else:
                self.filename_intensity[count] = \
                    str(this_exp['ophys_average_intensity_projection_image'])
                self.filename_segmask[count] = \
                    str(this_exp['max_int_mask_image'])
                ind = self.filename_intensity[count].find('ophys_experiment_')
                self.filename_exp_prefix[count] = \
                    self.filename_intensity[count][ind:ind + 26]
                count = count + 1

        logger.debug('Intensity images are: ', self.filename_intensity)
        logger.debug('Cell mask images are: ', self.filename_segmask)

    def parse_jsons(self, input_json):
        '''Parse input json file to genearte the
           necessary input files for nway cell matching.'''

        json_data = open(input_json).read()
        data = json.loads(json_data)

        self.args['output_directory'] = str(data['output_directory'])
        self.expnum = len(data['experiment_containers']['ophys_experiments'])

        self.filename_intensity = ["" for x in range(self.expnum)]
        self.filename_segmask = ["" for x in range(self.expnum)]
        self.filename_exp_prefix = ["" for x in range(self.expnum)]

        if self.expnum < 2:
            raise RuntimeError(
                    "There should be at least two "
                    "experiments! Check input json.")

        for i in range(self.expnum):

            this_exp = data['experiment_containers']['ophys_experiments'][i]

            # str() convert unicode string to regular string
            self.filename_intensity[i] = \
                str(
                    this_exp['ophys_average_intensity_projection_image'])
            self.filename_segmask[i] = str(this_exp['max_int_mask_image'])
            self.filename_exp_prefix[i] = re.findall(
                    self.args['id_pattern'], self.filename_intensity[i])[0]

        logger.debug('Intensity images are: ', self.filename_intensity)
        logger.debug('Cell mask images are: ', self.filename_segmask)

    def gen_label_roi_dict(self, filename_exp_prefix_fixed, input_json):
        '''Generate label to roi id mapping dictionary. ROIs are coded in input json.
           Labels are coded in segmented cell mask images.
        '''

        with open(input_json, 'r') as f:
            data = json.load(f)

        filename_segmask = os.path.join(
                self.args['output_directory'],
                filename_exp_prefix_fixed + '_maxInt_masks_relabel.tif')
        expnum = len(data['experiment_containers']['ophys_experiments'])

        for this_exp in data['experiment_containers']['ophys_experiments']:
            if filename_exp_prefix_fixed in this_exp['max_int_mask_image']:
                cell_rois = this_exp['cell_rois']
                break

        segmaskimg = read_tiff_3d(filename_segmask)
        segmaskimg = np.moveaxis(segmaskimg, 1, 2)

        cellnum = len(cell_rois)
        dict_label_to_roiid = dict()

        for cell_roi in cell_rois:
            x0 = cell_roi["x"]
            y0 = cell_roi["y"]
            w = cell_roi["width"]
            h = cell_roi["height"]
            # fancy indexing, indices of this sub-region
            coord = np.mgrid[x0:(x0 + w):1, y0:(y0 + h):1].reshape(2, -1).T

            # find unique non-zero labels in sub-region
            sub = segmaskimg[cell_roi["z"], coord[:, 0], coord[:, 1]]
            labels = np.delete(np.unique(sub), 0)

            # find label with largest area
            areas = np.array([np.count_nonzero(sub == l) for l in labels])
            mlabel = labels[np.argmax(areas)]

            dict_label_to_roiid[mlabel] = cell_roi["id"]

        mask_cellnum = segmaskimg.max()
        assert mask_cellnum == len(dict_label_to_roiid)

        return dict_label_to_roiid, mask_cellnum

    def gen_nway_table_with_redundancy(self):
        '''Generate initial Nway matching table with redundancy rows by
           scanning every matching pair in all pairwise matching results.
        '''

        matching_table_nway = []
        # counter for each pair-wise matching,
        # maximum value is C(N,2), N means Nway
        cnt = 0

        for i in range(self.expnum - 1):
            for j in range(i + 1, self.expnum):
                for line in self.matching_res_dict[cnt]['res']:
                    if (line[0] != -1) & (line[1] != -1):
                        this_record = np.zeros(self.expnum, dtype='int') - 1
                        this_record[i] = line[0]
                        this_record[j] = line[1]
                        matching_table_nway.append(this_record)
                cnt += 1

        return matching_table_nway

    @staticmethod
    def remove_nway_table_redundancy(table):
        '''Remove redundancy from the matching table. Redundancy include lines
           that are the same or one is the subset of another.
        '''
        # according to the note above, I would think this would work:
        #linenum, expnum = np.shape(table)
        #new_table = []
        #while len(new_table) != len(table):
        #    tups = []
        #    for ix in table:
        #        tups.append(set(
        #             [tuple((i, v)) for i, v in enumerate(ix) if v != -1]))

        #    for i, itup in enumerate(tups):
        #        for j, jtup in enumerate(tups):
        #            if j != i:
        #                if itup.intersection(jtup):
        #                    newline = np.ones_like(table[i]) * -1
        #                    ntup = itup.union(jtup)
        #                    [newline[n[0]] = n[



        # table = [v for i, v in enumerate(table) if keep[i]]
        # return table

        # but, it does not, so, I preserve the original logic for now
        # matching_table_nway = np.copy(matching_table_nway_ori)
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

        linenum = np.shape(matching_table_nway)[0]
        score = np.zeros(linenum)
        matching_table_nway_new = []

        # compute score for each record in matching_table_nway
        for i in range(linenum):

            score[i] = 0
            cnt = 0

            for j in range(self.expnum - 1):
                for k in range(j + 1, self.expnum):
                    [cellnum1, cellnum2] = np.shape(
                            self.matching_res_dict[cnt]['weight_matrix'])
                    if (
                            (matching_table_nway[i][j]-1 < cellnum1) and
                            (matching_table_nway[i][k]-1 < cellnum2)):
                        score[i] = \
                                score[i] + \
                                self.matching_res_dict[cnt]['weight_matrix'][
                                        matching_table_nway[i][j] - 1][
                                                matching_table_nway[i][k] - 1]
                        cnt = cnt + 1
            score[i] = score[i]/cnt

        # if two records share common elements in the
        # same column, then create an edge between them
        edge = np.zeros((linenum, linenum))
        node_exist = np.ones(linenum)

        for i in range(linenum - 1):  # start from the second line
            for j in range(i + 1, linenum):
                for k in range(self.expnum):

                    if (
                           (matching_table_nway[j][k] ==
                               matching_table_nway[i][k]) and
                           (matching_table_nway[i][k] != -1) and
                           (matching_table_nway[j][k] != -1)):
                        edge[i, j] = 1
                        break

        labelval = 0

        for i in range(linenum):

            if node_exist[i] == 1:
                idx = np.argwhere(edge[i, :] == 1)
                len_idx = len(idx)

                if len_idx == 0:  # no conflict with other matching
                    labelval = labelval + 1
                    this_record = np.append(matching_table_nway[i], labelval)
                    matching_table_nway_new.append(this_record)

                # score is the smallest, equal may lead
                # to conflict matching added to the list?
                elif score[i] <= np.min(score[idx]):
                    labelval = labelval + 1
                    this_record = np.append(matching_table_nway[i], labelval)
                    matching_table_nway_new.append(this_record.astype(int))

                    # remove the nodes connected to it
                    # as they have worse scores
                    edge[i, idx] = 0
                    for k in range(len_idx):
                        edge[idx[k], :] = 0
                    node_exist[idx] = 0

                else:  # prune the edge
                    edge[i, idx] = 0

        return matching_table_nway_new

    def add_remaining_cells(self, matching_table_nway):
        ''' Add remaining cells that do not find matches into the table.'''

        linenum = np.shape(matching_table_nway)[0]
        cnt = linenum
        matching_table_nway_new = matching_table_nway
        label_remain = []

        for j in range(self.expnum):
            labels = np.array(range(self.mask_cellnum[j])) + 1
            for i in range(linenum):
                labels = np.setdiff1d(labels, [matching_table_nway[i][j]])
            label_remain.append(labels)

        for i in range(self.expnum):
            num = len(label_remain[i])
            for j in range(num):
                cnt = cnt + 1
                this_record = np.zeros(self.expnum + 1) - 1
                this_record[i] = label_remain[i][j]
                this_record[self.expnum] = cnt
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

    def write_output_json(self, output_json):
        ''' Write the matching result into output json file. '''

        cellnum = np.shape(self.matching_table_nway)[0]
        matchingdata = dict()
        matchingdata["cell_rois"] = dict()
        prob = np.zeros(self.expnum)

        for i in range(cellnum):
            matching_exp_num = \
                    self.expnum - 1 - \
                    np.count_nonzero(self.matching_table_nway[i] == -1)
            prob[matching_exp_num] = prob[matching_exp_num] + 1
            logger.debug(self.matching_table_nway[i])

        for i in range(cellnum):
            labelstr = str(int(self.matching_table_nway[i][self.expnum]))
            thisrgn = []

            # Handle unsychronized roi and segmentation mask
            for j in range(self.expnum):
                if (
                    (int(self.matching_table_nway[i][j]) != -1) &
                    (
                        int(self.matching_table_nway[i][j]) in
                        self.dict_label_to_roiid[j].keys())):
                    thisrgn.append(
                            self.dict_label_to_roiid[j][
                                int(self.matching_table_nway[i][j])])

            matchingdata["cell_rois"][labelstr] = thisrgn

        matchingdata["transforms"] = []
        for k in self.matching_res_dict:
            matchingdata["transforms"].append({
                "moving": k['moving'],
                "fixed": k['fixed'],
                "transform": k['transform']})

        with open(output_json, 'w') as myfile:
            json.dump(matchingdata, myfile, sort_keys=True, indent=4)

    def write_output_images(self):
        ''' Write matching images. Matched cells are given the same label.'''

        for k in range(self.expnum):
            outimgfilename = os.path.join(
                self.args['output_directory'],
                self.filename_exp_prefix[k] + '_matching.tif')
            filename_segmask_relabel = os.path.join(
                self.args['output_directory'],
                self.filename_exp_prefix[k] + '_maxInt_masks_relabel.tif')
            segmask_3d = \
                read_tiff_3d(filename_segmask_relabel)

            matching_mask = np.zeros(segmask_3d.shape)
            linenum = np.shape(self.matching_table_nway)[0]

            for i in range(linenum):
                if self.matching_table_nway[i][k] > 0:
                    matching_mask[
                            segmask_3d == self.matching_table_nway[i][k]
                            ] = self.matching_table_nway[i][self.expnum]

            sitk_img = sitk.GetImageFromArray(matching_mask.astype(np.uint16))
            sitk.WriteImage(sitk_img, outimgfilename)

    def match_nway(self, para):
        '''Nway cell matching by calling pairwise
           matching and then combining the results'''

        # pair-wise matching
        self.matching_res_dict = []
        self.pair_matches = []
        for i in range(self.expnum - 1):
            pair_args = dict(self.args)
            pair_args['filename_intensity_fixed'] = self.filename_intensity[i]
            pair_args['filename_segmask_fixed'] = self.filename_segmask[i]
            pair_args['output_directory'] = self.args['output_directory']

            for j in range(i + 1, self.expnum):
                pair_args['filename_intensity_moving'] = \
                    self.filename_intensity[j]
                pair_args['filename_segmask_moving'] = \
                    self.filename_segmask[j]

                self.pair_matches.append(
                        PairwiseMatching(input_data=pair_args, args=[]))
                matching_pair = self.pair_matches[-1].run()

                self.matching_res_dict.append(matching_pair)

        # generate label id to roi id dictionary
        self.dict_label_to_roiid = []
        self.mask_cellnum = np.zeros(self.expnum, dtype=np.int)
        for i in range(self.expnum):
            this_dict_label_to_roiid, self.mask_cellnum[i] = \
                    self.gen_label_roi_dict(
                            self.filename_exp_prefix[i], para['input_json'])
            self.dict_label_to_roiid = np.append(
                    self.dict_label_to_roiid, this_dict_label_to_roiid)

        # generate N-way matching table
        matching_table_nway_tmp = self.gen_nway_table_with_redundancy()

        matching_table_nway_tmp = self.remove_nway_table_redundancy(
                matching_table_nway_tmp)
        logger.info('Pass.')

        logger.info('Pruning matching graph ...')
        matching_table_nway_tmp = self.prune_matching_graph(
                matching_table_nway_tmp)
        logger.info('Pass.')

        logger.info('Adding standalone cells ...')
        self.matching_table_nway = self.add_remaining_cells(
                matching_table_nway_tmp)
        logger.info('Pass.')

        return

    def run(self):
        ''' Main function of nway cell matching across multiple ophys sessions.
            The method takes three step:
            1) Image registration;
            2) Pairwise matching;
            3) Combining pairwise matching to generate Nway result.
        '''

        #self.args['output_directory'] = ""
        self.filename_intensity = []
        self.filename_segmask = []
        self.filename_exp_prefix = []
        self.expnum = 0
        self.matching_res_dict = []
        self.dict_label_to_roiid = []
        self.mask_cellnum = []
        self.matching_table_nway = []

        self.parse_jsons(self.args['input_json'])

        self.match_nway(self.args)
        logger.info("Nway matching is done!")

        self.write_matching_table()
        logger.info("Matching table is written!")

        self.write_output_json(self.args['output_json'])
        logger.info("Output json is generated!")

        self.write_output_images()
        logger.info("Matching images are written!")


if __name__ == "__main__":
    nmod = NwayMatching()
    nmod.run()
