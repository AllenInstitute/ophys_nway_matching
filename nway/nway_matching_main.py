# -*- coding: utf-8 -*-
'''
Main function of nway matching of cells

Copyright (c) Allen Institute for Brain Science

Usage:
--input_json /local1/fuhuil/work/data/ISI_and_Ophys/ophys_ophys/debug_Nway_matching/visbeh_10/visbeh_10_Slc17a7_VISp_175_input.json --output_json /local1/fuhuil/work/data/ISI_and_Ophys/ophys_ophys/debug_Nway_matching/visbeh_10/visbeh_10_Slc17a7_VISp_175_output.json --munkres_executable /local1/fuhuil/work/code/code_4_git/ophys_ophys_Nway/ophys_ophys_Nway/munkres/build/bp_matching

'''

import logging
import json
import numpy as np
import SimpleITK as sitk

from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import Boolean, Int, Str, Float

import pairwise_matching as pm
import region_properties as rp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CellMatchingParameters(ArgSchema):
    ''' Class that uses argschema to take care of input arguments '''

    save_registered_image = Boolean(required=False, default=True, description='Whether to save registered image.')
    maximum_distance = Int(required=False, default=10, description='Maximum distance (in pixels) between two cells, above which a match is always rejected.')
    diagnostic_figures = Boolean(required=False, default=False, desciption='Plot diagnostic figures.')
    registration_iterations = Int(required=False, default=1000, description='Number of iterations for intensity based registration')
    registration_precision = Float(required=False, default=1.5e-7, description='Threshold of squared error, below which registration is terminated')
    munkres_executable = Str(required=True, description='Executable of Kuhn-Munkres algorithm for bipartitite graph matching (with path information)')


def parse_command_line():
    ''' Parse command line using ArgSchemaParser to get input parameters. '''

    mod = ArgSchemaParser(schema_type=CellMatchingParameters)
    return mod.args


class NwayMatching(object):
    ''' Class for matching cells across arbitrary number of ophys sessions.

        Necessary files are obtained by parsing input json. Final Nway matching
        result is obtained by combining pairwise matching.
    '''

    def __init__(self):

        self.dir_output = ""
        self.filename_intensity = []
        self.filename_segmask = []
        self.filename_exp_prefix = []
        self.expnum = 0
        self.matching_res_dict = []
        self.dict_label_to_roiid = []
        self.mask_cellnum = []
        self.matching_table_nway = []

    def parse_jsons_with_reference_keyword(self, input_json):
        ''' Parse input json file to genearte the necessary input files for nway cell matching.

            This function deals with the old input json file format, with the
		  'reference_experiment' keyword.
        '''

        json_data = open(input_json).read()
        data = json.loads(json_data)

        self.dir_output = str(data['output_directory'])
        self.expnum = len(data['experiment_containers']['ophys_experiments'])

        self.filename_intensity = ["" for x in range(self.expnum)]
        self.filename_segmask = ["" for x in range(self.expnum)]
        self.filename_exp_prefix = ["" for x in range(self.expnum)]

        if self.expnum < 2:
            raise RuntimeError("There should be at least two experiments! Check input json. ")

        count = 1
        for i in range(self.expnum):

            this_exp = data['experiment_containers']['ophys_experiments'][i]

            reference_experiment = this_exp['reference_experiment']
            if str(reference_experiment).upper() == "TRUE":  # the reference is #0
                self.filename_intensity[0] = str(this_exp['ophys_average_intensity_projection_image'])  # str() convert unicode string to regular string
                self.filename_segmask[0] = str(this_exp['max_int_mask_image'])
                ind = self.filename_intensity[0].find('ophys_experiment_')
                self.filename_exp_prefix[0] = self.filename_intensity[0][ind:ind + 26]
            else:
                self.filename_intensity[count] = str(this_exp['ophys_average_intensity_projection_image'])
                self.filename_segmask[count] = str(this_exp['max_int_mask_image'])
                ind = self.filename_intensity[count].find('ophys_experiment_')
                self.filename_exp_prefix[count] = self.filename_intensity[count][ind:ind + 26]
                count = count + 1

        logger.debug('Intensity images are: ', self.filename_intensity)
        logger.debug('Cell mask images are: ', self.filename_segmask)

    def parse_jsons(self, input_json):
        '''Parse input json file to genearte the necessary input files for nway cell matching.'''

        json_data = open(input_json).read()
        data = json.loads(json_data)

        self.dir_output = str(data['output_directory'])
        self.expnum = len(data['experiment_containers']['ophys_experiments'])

        self.filename_intensity = ["" for x in range(self.expnum)]
        self.filename_segmask = ["" for x in range(self.expnum)]
        self.filename_exp_prefix = ["" for x in range(self.expnum)]

        if self.expnum < 2:
            raise RuntimeError("There should be at least two experiments! Check input json. ")

        for i in range(self.expnum):

            this_exp = data['experiment_containers']['ophys_experiments'][i]

            self.filename_intensity[i] = str(this_exp['ophys_average_intensity_projection_image'])  # str() convert unicode string to regular string
            self.filename_segmask[i] = str(this_exp['max_int_mask_image'])
            ind = self.filename_intensity[i].find('ophys_experiment_')
            self.filename_exp_prefix[i] = self.filename_intensity[i][ind:ind + 26]

        logger.debug('Intensity images are: ', self.filename_intensity)
        logger.debug('Cell mask images are: ', self.filename_segmask)

    def gen_label_roi_dict(self, filename_exp_prefix_fixed, input_json):
        '''Generate label to roi id mapping dictionary. ROIs are coded in input json.
           Labels are coded in segmented cell mask images.
        '''

        data = json.loads(open(input_json).read())
        filename_segmask = self.dir_output + filename_exp_prefix_fixed + '_maxInt_masks_relabel.tif'
        expnum = len(data['experiment_containers']['ophys_experiments'])

        for i in range(expnum):

            this_exp = data['experiment_containers']['ophys_experiments'][i]
            filename_max_int_mask_image = str(this_exp['max_int_mask_image'])
            ind = filename_max_int_mask_image.find(filename_exp_prefix_fixed)

            if ind != -1:
                cell_rois = this_exp['cell_rois']
                break


        img = sitk.ReadImage(filename_segmask)
        dim = img.GetDimension()

        if (dim != 2) and (dim != 3):
            raise RuntimeError("Image dimension must be 2 or 3! Check input json. ")

        if dim == 3:
            col, row, dep = img.GetSize()
            tmp = sitk.GetArrayFromImage(img)
        else:
            col, row = img.GetSize()
            dep = 1
            tmp = sitk.GetArrayFromImage(img)
            tmp = np.expand_dims(tmp, axis=0)

#        convert sitk array to numpy array
        segmaskimg = np.zeros((dep, col, row), dtype=np.int)

        for i in range(dep):
            segmaskimg[i, :, :] = tmp[i, :, :].T

        cellnum = len(cell_rois)
        dict_label_to_roiid = dict()

        for i in range(cellnum):
            x_start = int(cell_rois[i]["x"])
            y_start = int(cell_rois[i]["y"])
            z_start = int(cell_rois[i]["z"])
            width = int(cell_rois[i]["width"])
            height = int(cell_rois[i]["height"])

            # Compute coordinates of ROI pixels
            xscope = range(x_start, x_start + width)
            yscope = range(y_start, y_start + height)
            xscope_2d, yscope_2d = np.meshgrid(xscope, yscope)
            coord = np.vstack((xscope_2d.flatten(), yscope_2d.flatten()))
            coord = np.transpose(coord)

            thisframe = segmaskimg[z_start, :, :]
            labelvalue = np.unique(thisframe[coord[:, 0], coord[:, 1]])
            labelvalue = labelvalue[labelvalue > 0]

            labelnum = len(labelvalue)
            area = np.zeros((labelnum))
            maxarea = 0

            for j in range(labelnum):
                area[j] = np.count_nonzero(thisframe[coord[:, 0], coord[:, 1]] == labelvalue[j])
                if area[j] > maxarea:
                    maxarea = area[j]
                    idx = j

            dict_label_to_roiid[labelvalue[idx]] = cell_rois[i]["id"]

        segmaskimg2 = rp.RegionProperties(segmaskimg)
        labels_val, mask_cellnum = segmaskimg2.get_labels()

        return dict_label_to_roiid, mask_cellnum

    def gen_nway_table_with_redundancy(self):
        '''Generate initial Nway matching table with redundancy rows by
		 scanning every matching pair in all pairwise matching results.
        '''

        matching_table_nway = []
        cnt = 0 # counter for each pair-wise matching, maximum value is C(N,2), N means Nway

        for i in range(self.expnum - 1):
            for j in range(i + 1, self.expnum):

                linenum = np.shape(self.matching_res_dict[cnt]['res'])[0]

                for k in range(linenum): # scan each line in self.matching_res_dict[cnt]['res']

				# only consider lines that have cells matching to each other in session i and j
                    if (self.matching_res_dict[cnt]['res'][k, 0] != -1) and (self.matching_res_dict[cnt]['res'][k, 1] != -1):
                        this_record = np.zeros(self.expnum) - 1  # initiate every element to be -1, e.g., [-1, -1, -1, -1, -1]
                        this_record[i] = self.matching_res_dict[cnt]['res'][k, 0]
                        this_record[j] = self.matching_res_dict[cnt]['res'][k, 1]

                        matching_table_nway.append(this_record.astype(int)) # add the current line into nway table

                cnt = cnt + 1

        return matching_table_nway

    def remove_nway_table_redundancy(self, matching_table_nway_ori):
        '''Remove redundancy from the matching table. Redundancy include lines
           that are the same or one is the subset of another.
        '''

        matching_table_nway = np.copy(matching_table_nway_ori)
        linenum = np.shape(matching_table_nway)[0]
        stoptag = 0

        while stoptag == 0:
            matching_table_nway_new = [matching_table_nway[0]]

            for i in range(1, linenum):  # start from the second line
                num = np.shape(matching_table_nway_new)[0]

                for j in range(num):
                    mergetag = 1
                    postag = np.zeros(self.expnum, dtype=np.int)

                    for k in range(self.expnum):
                        if (matching_table_nway_new[j][k] != matching_table_nway[i][k]) and (
                                matching_table_nway[i][k] != -1) and (matching_table_nway_new[j][k] != -1):
                            mergetag = 0
                            break
                        if (matching_table_nway_new[j][k] == -1) or (matching_table_nway[i][k] == -1):
                            postag[k] = -1

                    if np.sum(postag) == -self.expnum:
                        mergetag = 0

                    if mergetag == 1:

                        for k in range(self.expnum):
                            if matching_table_nway[i][k] != -1:
                                matching_table_nway_new[j][k] = matching_table_nway[i][k]
                        break  # break for loop j, only merge to one previous record

                if mergetag == 0:
                    matching_table_nway_new.append(matching_table_nway[i])

            linenum_new = np.shape(matching_table_nway_new)[0]

            if linenum == linenum_new:
                stoptag = 1
            else:
                matching_table_nway = np.copy(matching_table_nway_new)
                linenum = linenum_new

        return matching_table_nway_new

    def prune_matching_graph(self, matching_table_nway):
        ''' Prune matching graph to remove matching conflicts.

            This is achieved by creating a graph. Each matching line is a node in the graph.
            An edge indicates conflict between matchings respresented by the two nodes. The
            final result is obtained by pruning the graph based on weight matrix computed earlier.

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
                    [cellnum1, cellnum2] = np.shape(self.matching_res_dict[cnt]['weight_matrix'])
                    if (matching_table_nway[i][j]-1 < cellnum1) and (matching_table_nway[i][k]-1 < cellnum2):
                        score[i] = score[i] + self.matching_res_dict[cnt]['weight_matrix'][matching_table_nway[i][j] - 1][matching_table_nway[i][k] - 1]
                        cnt = cnt + 1
            score[i] = score[i]/cnt

        # if two records share common elements in the same column, then create an edge between them
        edge = np.zeros((linenum, linenum))
        node_exist = np.ones(linenum)

        for i in range(linenum - 1):  # start from the second line
            for j in range(i + 1, linenum):
                for k in range(self.expnum):

                    if (matching_table_nway[j][k] == matching_table_nway[i][k]) and (
                            matching_table_nway[i][k] != -1) and (matching_table_nway[j][k] != -1):
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

                elif score[i] <= np.min(score[idx]):  # score is the smallest, equal may lead to conflict matching added to the list?
                    labelval = labelval + 1
                    this_record = np.append(matching_table_nway[i], labelval)
                    matching_table_nway_new.append(this_record.astype(int))

                    # remove the nodes connected to it, as they have worse scores
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
		  The last entry is the unified label of the cells.

	   '''
        filename_matching_table = self.dir_output + 'matching_result.txt'
        np.savetxt(filename_matching_table, self.matching_table_nway, delimiter=' ', fmt='%d')

    def write_output_json(self, output_json):
        ''' Write the matching result into output json file. '''

        cellnum = np.shape(self.matching_table_nway)[0]
        matchingdata = dict()
        matchingdata["cell_rois"] = dict()
        prob = np.zeros(self.expnum)

        for i in range(cellnum):
            matching_exp_num = self.expnum - 1 - np.count_nonzero(self.matching_table_nway[i] == -1)
            prob[matching_exp_num] = prob[matching_exp_num] + 1
            logger.debug(self.matching_table_nway[i])

        for i in range(cellnum):
            labelstr = str(int(self.matching_table_nway[i][self.expnum]))
            thisrgn = []

            # Handle unsychronized roi and segmentation mask
            for j in range(self.expnum):
                if (int(self.matching_table_nway[i][j]) != -1) & (int(self.matching_table_nway[i][j]) in self.dict_label_to_roiid[j].keys()):
                    thisrgn.append(self.dict_label_to_roiid[j][int(self.matching_table_nway[i][j])])

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
            outimgfilename = self.dir_output + self.filename_exp_prefix[k] + '_matching.tif'
            filename_segmask_relabel = self.dir_output + self.filename_exp_prefix[k] + '_maxInt_masks_relabel.tif'
            segmask_3d, col_segmask, row_segmask, dep_segmask = pm.read_tiff_3d(filename_segmask_relabel)

            # switch row and col of segmasks
            segmask_3d_tmp = np.zeros((dep_segmask, row_segmask, col_segmask), dtype=np.int)
            for i in range(dep_segmask):
                segmask_3d_tmp[i, :, :] = np.transpose(segmask_3d[i, :, :])
            segmask_3d = np.copy(segmask_3d_tmp)

            matching_mask = np.zeros((dep_segmask, row_segmask, col_segmask))
            linenum = np.shape(self.matching_table_nway)[0]

            for i in range(linenum):
                if self.matching_table_nway[i][k] > 0:
                    matching_mask[segmask_3d == self.matching_table_nway[i][k]] = self.matching_table_nway[i][self.expnum]

            sitk_img = sitk.GetImageFromArray(matching_mask.astype(np.uint16))
            sitk.WriteImage(sitk_img, outimgfilename)

    def match_nway(self, para):
        ''' Nway cell matching by calling pairwise matching and then combining the results '''

        # pair-wise matching
        self.matching_res_dict = []

        for i in range(self.expnum - 1):

            para_matching = dict()
            para_matching['filename_intensity_fixed'] = self.filename_intensity[i]
            para_matching['filename_segmask_fixed'] = self.filename_segmask[i]

            ind = self.filename_intensity[i].find('ophys_experiment_')
            para_matching['filename_exp_prefix_fixed'] = self.filename_intensity[i][ind:ind + 26]

            for j in range(i + 1, self.expnum):

                para_matching['filename_intensity_moving'] = self.filename_intensity[j]
                para_matching['filename_segmask_moving'] = self.filename_segmask[j]

                logger.info('Matching %s against %s ...', para_matching['filename_intensity_moving'], para_matching['filename_intensity_fixed'])
                matching = pm.ComputePairWiseMatch(self.dir_output)

                ind = self.filename_intensity[j].find('ophys_experiment_')
                para_matching['filename_exp_prefix_moving'] = self.filename_intensity[j][ind:ind + 26]

                matching_pair = matching.match_pairs(para, para_matching)

                self.matching_res_dict.append(matching_pair)

        # generate label id to roi id dictionary
        self.dict_label_to_roiid = []
        self.mask_cellnum = np.zeros(self.expnum, dtype=np.int)

        for i in range(self.expnum):
            this_dict_label_to_roiid, self.mask_cellnum[i] = self.gen_label_roi_dict(self.filename_exp_prefix[i], para['input_json'])
            self.dict_label_to_roiid = np.append(self.dict_label_to_roiid, this_dict_label_to_roiid)

        # generate N-way matching table
        logger.info('Generating Nway matching table with redundancy ...')
        matching_table_nway_tmp = self.gen_nway_table_with_redundancy()
        logger.info('Pass.')

        logger.info('Removing redunant inforamtion from Nway matching table...')
        matching_table_nway_tmp = self.remove_nway_table_redundancy(matching_table_nway_tmp)
        logger.info('Pass.')

        logger.info('Pruning matching graph ...')
        matching_table_nway_tmp = self.prune_matching_graph(matching_table_nway_tmp)
        logger.info('Pass.')

        logger.info('Adding standalone cells ...')
        self.matching_table_nway = self.add_remaining_cells(matching_table_nway_tmp)
        logger.info('Pass.')

        return


def main():
    ''' Main function of nway cell matching across multiple ophys sessions.

        The method takes three step:
        1) Image registration;
        2) Pairwise matching;
        3) Combining pairwise matching to generate Nway result.
    '''

    para = parse_command_line()

    trial = NwayMatching()

    trial.parse_jsons(para['input_json']) # call parse_jsons_with_reference_keyword() for old json format
    logger.info('Parsing input json is done!')

    trial.match_nway(para)
    logger.info("Nway matching is done!")

    trial.write_matching_table()
    logger.info("Matching table is written!")

    trial.write_output_json(para['output_json'])
    logger.info("Output json is generated!")

    trial.write_output_images()
    logger.info("Matching images are written!")


if __name__ == "__main__":

    main()
