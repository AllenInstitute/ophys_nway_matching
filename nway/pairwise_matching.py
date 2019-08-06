# -*- coding: utf-8 -*-
"""

Pair-wise matching called by nway matching main function

Copyright (c) Allen Institute for Brain Science
"""
import os
import logging
import subprocess
import shlex
import numpy as np
from PIL import Image as im
from skimage import measure as ms
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2

import region_properties as rp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

plt.ion()


def sitk2numpyarray(array_sitk):
    ''' Convert sitk array to numpy array. '''

    dim = np.shape(array_sitk)

    if len(dim) == 3:
        dep, col, row = np.shape(array_sitk)
        array_np = np.zeros((dep, row, col), dtype=np.int)
        for i in range(dep):
            array_np[i, :, :] = array_sitk[i, :, :].T
    elif len(dim) == 2:
        array_np = array_sitk.T
    else:
        logger.error("Error in converting sitk array to numpy array. Image should be either 2D or 3D.")
        return

    return array_np


def imshow3d(img3d, display_row_num, display_col_num):
    '''Display 3d images.'''

    dep = np.shape(img3d)[0]
    if display_row_num * display_col_num < dep:
        logger.error("The number of row to display times the number of columns is less than the depth of the image. Fail to display.")
        return

    figs = plt.figure()

    str_row = str(display_row_num)
    str_col = str(display_col_num)

    for i in range(dep):
        id_panel = str_row + str_col + str(i + 1)
        figs.add_subplot(id_panel)
        plt.imshow(img3d[i, :, :])


def read_tiff_3d(filename):
    '''Read 3d tiff files. '''

    img = sitk.ReadImage(filename)
    dim = img.GetDimension()

    if (dim != 2) and (dim != 3):
        logger.error("Error in reading 3D tiff. Image dimension must be 2 or 3! ")

    if dim == 3:
        col, row, dep = img.GetSize()
        tmp = sitk.GetArrayFromImage(img)
        img3d = sitk2numpyarray(tmp)  # convert sitk array to numpy array
    else:
        col, row = img.GetSize()
        dep = 1
        tmp = sitk.GetArrayFromImage(img)
        img2d = sitk2numpyarray(tmp)  # convert sitk array to numpy array
        img3d = np.expand_dims(img2d, axis=0)

    return img3d, col, row, dep


def relabel(maskimg_3d):
    ''' Relabel mask image to make labels continous and unique'''

    num_images = np.shape(maskimg_3d)[0]
    labeloffset = 0

    for k in range(0, num_images):
        labelimg = ms.label((maskimg_3d[k, :, :] > 0))
        labelimg[labelimg > 0] = labelimg[labelimg > 0] + labeloffset
        maskimg_3d[k, :, :] = labelimg
        labeloffset = np.amax(labelimg)

    return maskimg_3d


def run_bipartite_matching(tool_name_args):
    ''' call c++ executable.'''

    args = shlex.split(tool_name_args[0])
    subprocess.check_call(args)


def remove_tmp_files(filename):
    ''' Remove temporary files. '''

    if os.path.isfile(filename):
        os.remove(filename)


class ComputePairWiseMatch(object):
    ''' Class for pairwise ophys matching.

        Images are registered using intensity correlation.
        Best matches between cells are determined by bipartite graph matching.
    '''

    def __init__(self, dir_output):

        self.dir_output = dir_output
        self.segmask_fixed_3d = np.array([])
        self.segmask_moving_3d = np.array([])
        self.segmask_moving_3d_registered = np.array([])
        self.matching_table = np.array([])
        self.segmask_fixed_3d_sz = 0
        self.segmask_moving_3d_sz = 0

    def match_pairs(self, para, para_matching):
        ''' Pairwise matching of ophys-ophys sessions. '''


        filename_output_fixed_matched_img = ''
        filename_output_moving_matched_img = ''

        filename_matching_table = para_matching['filename_exp_prefix_fixed'] + '_to_' + para_matching['filename_exp_prefix_moving']

        # register the average intensity images
        tform = self.register_intensity_images(para, para_matching['filename_intensity_fixed'], para_matching['filename_intensity_moving'])

        # read segmentation masks
        self.segmask_fixed_3d, col_segmask_fixed_3d, row_segmask_fixed_3d, dep_segmask_fixed_3d = read_tiff_3d(para_matching['filename_segmask_fixed'])
        self.segmask_moving_3d, col_segmask_moving_3d, row_segmask_moving_3d, dep_segmask_moving_3d = read_tiff_3d(para_matching['filename_segmask_moving'])

        # switch row and col of segmasks
        segmask_fixed_3d_tmp = np.zeros(
            (dep_segmask_fixed_3d,
             row_segmask_fixed_3d,
             col_segmask_fixed_3d),
            dtype=np.int)

        for i in range(dep_segmask_fixed_3d):
            segmask_fixed_3d_tmp[i, :, :] = np.transpose(self.segmask_fixed_3d[i, :, :])
        self.segmask_fixed_3d = np.copy(segmask_fixed_3d_tmp)

        segmask_moving_3d_tmp = np.zeros((dep_segmask_moving_3d, row_segmask_moving_3d, col_segmask_moving_3d))
        for i in range(dep_segmask_moving_3d):
            segmask_moving_3d_tmp[i, :, :] = np.transpose(self.segmask_moving_3d[i, :, :])
        self.segmask_moving_3d = np.copy(segmask_moving_3d_tmp)

        self.segmask_fixed_3d_sz = [
            dep_segmask_fixed_3d,
            row_segmask_fixed_3d,
            col_segmask_fixed_3d]
        self.segmask_moving_3d_sz = [
            dep_segmask_moving_3d,
            row_segmask_moving_3d,
            col_segmask_moving_3d]

        # relabeling segmentation masks because they are unit 8 data type and only code 255 regions
        self.segmask_fixed_3d = relabel(self.segmask_fixed_3d)
        self.segmask_moving_3d = relabel(self.segmask_moving_3d)

        filename_segmask_fixed_relabel = self.dir_output + para_matching['filename_exp_prefix_fixed'] + '_maxInt_masks_relabel.tif'

        sitk_segmask_fixed = sitk.GetImageFromArray(self.segmask_fixed_3d.astype(np.uint16))
        sitk.WriteImage(sitk_segmask_fixed, filename_segmask_fixed_relabel)

        filename_segmask_moving_relabel = self.dir_output + para_matching['filename_exp_prefix_moving'] + '_maxInt_masks_relabel.tif'

        sitk_segmask_moving = sitk.GetImageFromArray(self.segmask_moving_3d.astype(np.uint16))
        sitk.WriteImage(sitk_segmask_moving, filename_segmask_moving_relabel)

        if para['diagnostic_figures'] == 1:
            imshow3d(self.segmask_fixed_3d, 1, self.segmask_fixed_3d_sz[0])
            imshow3d(self.segmask_moving_3d, 1, self.segmask_moving_3d_sz[0])

        # transform moving segmentation masks
        self.register_mask_images(tform)  # compute self.segmask_moving_3d_registered

        if para['diagnostic_figures'] == 1:
            imshow3d(self.segmask_moving_3d_registered, 1, self.segmask_moving_3d_sz[0])

        # matching cells
        tmp_filenames = dict()
        tmp_filenames['fixed_matched_img'] = filename_output_fixed_matched_img
        tmp_filenames['moving_matched_img'] = filename_output_moving_matched_img
        tmp_filenames['matching_table'] = filename_matching_table

        [matching_ratio_fixed, matching_ratio_moving, weight_matrix] = self.cell_matching(para, tmp_filenames)

        matching_pairs = dict()
        matching_pairs['res'] = self.matching_table
        matching_pairs['mr_i'] = matching_ratio_fixed
        matching_pairs['mr_j'] = matching_ratio_moving
        matching_pairs['segmask_i'] = self.segmask_fixed_3d
        matching_pairs['segmask_j'] = self.segmask_moving_3d_registered
        matching_pairs['weight_matrix'] = weight_matrix
        matching_pairs['moving'] = para_matching['filename_intensity_moving']
        matching_pairs['fixed'] = para_matching['filename_intensity_fixed']
        matching_pairs['transform'] = tform.tolist()

        return matching_pairs


    def register_intensity_images(self, para, filename_int_fixed, filename_int_moving):
        ''' Register the average intensity images of the two ophys sessions using affine transformation'''

        # read average intensity images
        img_fixed = np.array(im.open(filename_int_fixed))
        img_moving = np.array(im.open(filename_int_moving))

        # Find size of img_fixed
        sz_fixed = np.shape(img_fixed)

        # Define the motion model
        warp_mode = cv2.MOTION_AFFINE

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            tform = np.eye(3, 3, dtype=np.float32)
        else:
            tform = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    para['registration_iterations'], para['registration_precision'])

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (ccval, tform) = cv2.findTransformECC(img_fixed,
                                              img_moving, tform, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            img_moving_registered = cv2.warpPerspective(
                img_moving,
                tform,
                (sz_fixed[1],
                 sz_fixed[0]),
                flags=cv2.INTER_LINEAR +
                cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            img_moving_registered = cv2.warpAffine(
                img_moving,
                tform,
                (sz_fixed[1],
                 sz_fixed[0]),
                flags=cv2.INTER_LINEAR +
                cv2.WARP_INVERSE_MAP)

        if para['diagnostic_figures'] == 1:

            sz0, sz1 = np.shape(img_fixed)
            img_overlay_ori = np.zeros((sz0, sz1, 3), np.uint8)
            img_overlay = np.zeros((sz0, sz1, 3), np.uint8)

            img_overlay_ori[:, :, 0] = img_fixed
            img_overlay_ori[:, :, 1] = img_moving

            img_overlay[:, :, 0] = img_fixed
            img_overlay[:, :, 1] = img_moving_registered

            figs = plt.figure()
            figs.add_subplot(231)
            plt.imshow(img_fixed, cmap='gray')
            figs.add_subplot(232)
            plt.imshow(img_moving, cmap='gray')
            figs.add_subplot(233)
            plt.imshow(img_moving_registered, cmap='gray')
            figs.add_subplot(234)
            plt.imshow(img_overlay_ori)
            figs.add_subplot(235)
            plt.imshow(img_overlay)

        if para['save_registered_image'] == 1:

            ind_moving = filename_int_moving.find('ophys_experiment_')
            ind_fixed = filename_int_fixed.find('ophys_experiment_')

            filename_prefix_moving = filename_int_moving[ind_moving:ind_moving + 26]
            filename_prefix_fixed = filename_int_fixed[ind_fixed:ind_fixed + 26]

            filename = self.dir_output + 'register_' + \
                filename_prefix_moving + '_to_' + filename_prefix_fixed + '.tif'
            sitk_img_moving_registered = sitk.GetImageFromArray(
                img_moving_registered.astype(np.uint8))
            sitk.WriteImage(sitk_img_moving_registered, filename)

        return tform

    def register_mask_images(self, tform):
        ''' Transform segmentation masks using the affine transformation defined by tform'''

        row_segmask_fixed_3d = self.segmask_fixed_3d_sz[1]
        col_segmask_fixed_3d = self.segmask_fixed_3d_sz[2]
        num_images_moving = self.segmask_moving_3d_sz[0]

        self.segmask_moving_3d_registered = np.zeros((num_images_moving, row_segmask_fixed_3d, col_segmask_fixed_3d), dtype=np.int)

        for k in range(num_images_moving):  # number of frames in segmask_moving_3d

            num = np.int(np.amax(self.segmask_moving_3d[k, :, :]))
            segmask_2d_registered = np.zeros((row_segmask_fixed_3d, col_segmask_fixed_3d))

            for i in range(1, num + 1):  # number of cells in the current frame

#                tmp = np.zeros((row_segmask_fixed_3d, col_segmask_fixed_3d))
                tmp = np.zeros((self.segmask_moving_3d_sz[1], self.segmask_moving_3d_sz[2]))
                tmp[self.segmask_moving_3d[k, :, :] == i] = i

                if np.count_nonzero(tmp) > 0:  # this label exists

                    tmp_registered = cv2.warpAffine(
                        tmp,
                        tform,
                        (col_segmask_fixed_3d,
                         row_segmask_fixed_3d),
                        flags=cv2.INTER_LINEAR +
                        cv2.WARP_INVERSE_MAP)

                    segmask_2d_registered[tmp_registered > 0] = i

            self.segmask_moving_3d_registered[k, :, :] = segmask_2d_registered

        np.amax(self.segmask_moving_3d_registered)

        return self.segmask_moving_3d_registered

    def compute_features(self, para, filename_weightmatrix):
        ''' Computer features and weight matrix needed for bipartite graph matching.'''

        # compute features of fixed and moving images
        fixed_img = rp.RegionProperties(self.segmask_fixed_3d)
        moving_img = rp.RegionProperties(self.segmask_moving_3d_registered)

        fixed_fea = dict()
        moving_fea = dict()

        fixed_fea['areas'] = fixed_img.get_areas()
        moving_fea['areas'] = moving_img.get_areas()

        fixed_fea['labels'], fixed_fea['cellnum'] = fixed_img.get_labels()
        moving_fea['labels'], moving_fea['cellnum'] = moving_img.get_labels()

        fixed_fea['centers'] = fixed_img.get_centers()
        moving_fea['centers'] = moving_img.get_centers()

        fixed_fea['centers'] = fixed_fea['centers'][:, 1:3]  # take only x,y value
        moving_fea['centers'] = moving_fea['centers'][:, 1:3]  # take only x,y value

        pixelidxlist_fixed = fixed_img.get_pixel_idx_list()     # dictionary

        [fixed_fea['num_img'], fixed_fea['row_segmask'], fixed_fea['col_segmask']] = self.segmask_fixed_3d_sz
        [moving_fea['num_img'], moving_fea['row_segmask'], moving_fea['col_segmask']] = self.segmask_moving_3d_sz

        # compute features of bipartite graph edges
        edge_fea = dict()
        edge_fea['dist'] = np.zeros((fixed_fea['cellnum'], moving_fea['cellnum']))
        edge_fea['overlap'] = np.zeros((fixed_fea['cellnum'], moving_fea['cellnum']))

        for i in range(0, fixed_fea['cellnum']):
            # compute pair-wise distances between cell centers
            for j in range(0, moving_fea['cellnum']):
                edge_fea['dist'][i, j] = np.sqrt((fixed_fea['centers'][i, 0] - moving_fea['centers'][j, 0])
                                                 * (fixed_fea['centers'][i, 0] - moving_fea['centers'][j, 0])
                                                 + (fixed_fea['centers'][i, 1] - moving_fea['centers'][j, 1])
                                                 * (fixed_fea['centers'][i, 1] - moving_fea['centers'][j, 1]))

            segmask_fixed_this_layer = self.segmask_fixed_3d[pixelidxlist_fixed[fixed_fea['labels'][i]][0, 0], :, :]
            cellmask_fixed = (segmask_fixed_this_layer == i + 1)

            # compute overlap between cell centers
            for j in range(0, moving_fea['num_img']):

                segmask_moving_this_layer = self.segmask_moving_3d_registered[j, :, :]

                labels_moving_this_layer = np.unique(segmask_moving_this_layer[cellmask_fixed>0]) -[0]  # exclude background with label 0
                labels_moving_this_layer_num = len(labels_moving_this_layer)

                for k in range(0, labels_moving_this_layer_num): # Compute for each overlapping cell in moving image

                    cellmask_moving = (segmask_moving_this_layer == labels_moving_this_layer[k])

                    mask_intersection = np.multiply(cellmask_moving, cellmask_fixed)
                    mask_union = np.add(cellmask_moving, cellmask_fixed)
                    area_intersection = np.count_nonzero(mask_intersection)
                    area_union = np.count_nonzero(mask_union)
                    edge_fea['overlap'][i, labels_moving_this_layer[k] - 1] = area_intersection / area_union #IoU

        dist2 = np.copy(edge_fea['dist'])
        dist2 = dist2 / para['maximum_distance']
        dist2[dist2 > 1] = 999.0
#        overlap2 = 1 - edge_fea['overlap'].astype(float) / np.amax(edge_fea['overlap'])
        overlap2 = 1 - edge_fea['overlap'].astype(float)
        edge_fea['weight_matrix'] = dist2 + overlap2

        if para['diagnostic_figures'] == 1:
            figs = plt.figure()
            figs.add_subplot(221)
            plt.imshow(dist2, cmap='gray')
            figs.add_subplot(222)
            plt.imshow(overlap2, cmap='gray')
            figs.add_subplot(223)
            plt.imshow(edge_fea['weight_matrix'], cmap='gray')

        np.savetxt(filename_weightmatrix, edge_fea['weight_matrix'], delimiter=' ')

        return fixed_fea, moving_fea, edge_fea

    def write_matching_images(self, para):
        ''' Write matching results into images. The same label codes the same cell.'''

        if (len(para['filename_fixed']) > 0) & (len(para['filename_moving']) > 0):

            para['filename_fixed'] = self.dir_output + para['filename_fixed']
            para['filename_moving'] = self.dir_output + para['filename_moving']

            segmask_moving_3d_matching = np.zeros(self.segmask_moving_3d_registered.shape)
            segmask_fixed_3d_matching = np.zeros(self.segmask_fixed_3d.shape)

            for i in range(self.matching_table.shape[0]):
                if (self.matching_table[i, 0] > 0 and self.matching_table[i, 1] > 0):

                    segmask_fixed_3d_matching[self.segmask_fixed_3d == para['labels_fixed'][self.matching_table[i, 0] - 1]] = self.matching_table[i, 0]
                    segmask_moving_3d_matching[self.segmask_moving_3d_registered == para['labels_moving'][self.matching_table[i, 1] - 1]] = self.matching_table[i, 0]

            sitk_img1 = sitk.GetImageFromArray(segmask_fixed_3d_matching.astype(np.uint16))
            sitk.WriteImage(sitk_img1, para['filename_fixed'])

            sitk_img2 = sitk.GetImageFromArray(segmask_moving_3d_matching.astype(np.uint16))
            sitk.WriteImage(sitk_img2, para['filename_moving'])

    def gen_matching_table(self, para, para_matching):
        '''Generate self.matching_table using bipartite graph matching.'''

        tool_name_args = [para['munkres_executable'] + " " + \
                         para_matching['filename_weightmatrix'] + " " + \
                         str(para_matching['fixed_cellnum']) + " " + \
                         str(para_matching['moving_cellnum']) + " " + \
                         para_matching['filename_tmpmatching']]

        run_bipartite_matching(tool_name_args)

        # load matching result produced by bp_matching
        matching_pair = np.loadtxt(para_matching['filename_tmpmatching'], delimiter=' ').astype(int)

        # remove pairs that do not satisfy distance condition
        # matching_num is the smaller value of the number of regions in fixed
        # or moving masks
        num_matched = 0

        # determine the number of rows in matching_table
        count = para_matching['fixed_cellnum']  # include all cells in fixed image

        for i in range(para_matching['moving_cellnum']):
            ind = np.argwhere(matching_pair[:, 1] == i)
            if ind.size == 0:
                count = count + 1 # add those moving cells that are not in matching_pair
            elif para_matching['edge_fea']['dist'][int(matching_pair[ind[0][0], 0]), \
                                                   int(matching_pair[ind[0][0], 1])] >= para['maximum_distance']:
			# revent cells too far from matching even if matching_pair match them
                count = count + 1

        self.matching_table = np.zeros((count, 5))

        # add all cells in fixed image
        for i in range(para_matching['fixed_cellnum']):
            ind = np.argwhere(matching_pair[:, 0] == i)
            if ind.size != 0:
                if para_matching['edge_fea']['dist'][int(matching_pair[ind[0], 0][0]), int(
                        matching_pair[ind[0], 1][0])] < para['maximum_distance']:
                    # label index starts from 0, but label value starts from 1,
                    self.matching_table[i, 0] = matching_pair[ind, 0] + 1
                    self.matching_table[i, 1] = matching_pair[ind, 1] + 1
                    self.matching_table[i, 2] = para_matching['edge_fea']['dist'][int(matching_pair[ind[0], 0][0]), int(matching_pair[ind[0], 1][0])]
                    self.matching_table[i, 3] = para_matching['edge_fea']['overlap'][int(matching_pair[ind[0], 0][0]), int(matching_pair[ind[0], 1][0])]
                    self.matching_table[i, 4] = para_matching['edge_fea']['weight_matrix'][int(matching_pair[ind[0], 0][0]), int(matching_pair[ind[0], 1][0])]
                    num_matched = num_matched + 1
                else: #does not allow matching
                    # label index starts from 0, but label value starts from 1,
                    self.matching_table[i, 0] = matching_pair[ind, 0] + 1
                    self.matching_table[i, 1] = -1
                    self.matching_table[i, 2] = -1
                    self.matching_table[i, 3] = -1
                    self.matching_table[i, 4] = -1
            else:
                # label index starts from 0, but label value starts from 1,
                self.matching_table[i, 0] = i + 1
                self.matching_table[i, 1] = -1
                self.matching_table[i, 2] = -1
                self.matching_table[i, 3] = -1
                self.matching_table[i, 4] = -1

        # add the remaining cells in moving image
        count = para_matching['fixed_cellnum']

        for i in range(para_matching['moving_cellnum']):
            ind = np.argwhere(self.matching_table[:, 1] == i + 1) # notice self.matching_table, not matching_pair
            if ind.size == 0:
                # label index starts from 0, but label value starts from 1,
                self.matching_table[count, 0] = -1
                self.matching_table[count, 1] = i + 1
                self.matching_table[count, 2] = -1
                self.matching_table[count, 3] = -1
                self.matching_table[count, 4] = -1
                count = count + 1

        return num_matched


    def cell_matching(self, para, tmp_filenames):
        ''' Function that matches cells. '''

	   # compute features
        filename_weightmatrix = self.dir_output + 'weightmatrix.txt'
        fixed_fea, moving_fea, edge_fea = self.compute_features(para, filename_weightmatrix)

        # Generate matching result by calling bipartite graph matching in c++
        filename_tmpmatching = self.dir_output + 'matching_result_temporary.txt'

        para_matching = dict()

        para_matching['filename_weightmatrix'] = filename_weightmatrix
        para_matching['filename_tmpmatching'] = filename_tmpmatching
        para_matching['fixed_cellnum'] = fixed_fea['cellnum']
        para_matching['moving_cellnum'] = moving_fea['cellnum']
        para_matching['edge_fea'] = edge_fea

        num_matched = self.gen_matching_table(para, para_matching)

        # write matching images, matched cells are given the same label, unmatched cells are not labeled in the image
        para_w = dict()
        para_w['filename_fixed'] = tmp_filenames['fixed_matched_img']
        para_w['filename_moving'] = tmp_filenames['moving_matched_img']
        para_w['labels_fixed'] = fixed_fea['labels']
        para_w['labels_moving'] = moving_fea['labels']

        self.write_matching_images(para_w)

        # write matching table
        if len(tmp_filenames['matching_table']) > 0:
            filename_matching_table = self.dir_output + tmp_filenames['matching_table']
            np.savetxt(filename_matching_table, self.matching_table, delimiter=' ', fmt="%d %d %7.4e %7.4e %7.4e")

        matching_ratio_fixed = np.float(num_matched) / fixed_fea['cellnum']
        matching_ratio_moving = np.float(num_matched) / moving_fea['cellnum']

        # remove temporary files
        remove_tmp_files(filename_weightmatrix)
        remove_tmp_files(filename_tmpmatching)

        return matching_ratio_fixed, matching_ratio_moving, edge_fea['weight_matrix']
