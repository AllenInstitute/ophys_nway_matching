# -*- coding: utf-8 -*-
"""

Pair-wise matching called by nway matching main function

Copyright (c) Allen Institute for Brain Science
"""
import os
import re
import logging
import subprocess
import shlex
import numpy as np
from PIL import Image as im
from skimage import measure as ms
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import scipy.spatial
import scipy.optimize
from argschema import ArgSchemaParser

from nway.schemas import PairwiseMatchingSchema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

plt.ion()


def region_properties(mask):
    if len(mask.shape) not in [2, 3]:
        raise ValueError("region_properties needs a 2D or 3D image")

    prop = {}
    prop['labels'] = np.arange(mask.max()) + 1
    n = len(prop['labels'])
    prop['centers'] = np.zeros((n, 2))
    prop['pixels'] = [[]] * n
    for i, label in enumerate(prop['labels']):
        label_locs = np.argwhere(mask == label)
        if len(label_locs) > 0:
            # the rounding int cast here should not be here
            # leaving it in place for now to preserve legacy
            # matching
            prop['centers'][i] = np.round(
                    label_locs.mean(axis=0)).astype('int')[[-1, -2]]
        prop['pixels'][i] = set([tuple(x) for x in label_locs])
    return prop


def calculate_cost_matrix(mask1, mask2, maximum_distance):
    ''' Computer features and weight matrix needed for
        bipartite graph matching.'''

    prop1 = region_properties(mask1)
    prop2 = region_properties(mask2)

    # compute features of bipartite graph edges
    overlap = np.zeros((
            prop1['labels'].size,
            prop2['labels'].size))

    # compute pair-wise distances between cell centers
    distance = scipy.spatial.distance.cdist(
            prop1['centers'], prop2['centers'], 'euclidean')

    for i, ipix in enumerate(prop1['pixels']):
        for j, jpix in enumerate(prop2['pixels']):
            overlap[i, j] = \
                    len(ipix.intersection(jpix)) / len(ipix.union(jpix))

    dist2 = distance / maximum_distance
    dist2[dist2 > 1] = 999.0
    cost_matrix = dist2 + (1.0 - overlap)

    return prop1, prop2, distance, overlap, cost_matrix


def transform_masks(moving, dst_shape, tform):
    ''' Transform segmentation masks using the affine
        transformation defined by tform'''

    transformed_3d = np.zeros(
            (
                moving.shape[0],
                dst_shape[0],
                dst_shape[1]),
            dtype=np.int)

    for k, frame in enumerate(moving):
        labels = np.unique(frame)
        transformed_2d = np.zeros(dst_shape)

        for label in labels:
            tmp = np.zeros_like(frame).astype('float32')
            tmp[frame == label] = 1
            tmp_registered = cv2.warpAffine(
                    tmp,
                    tform,
                    dst_shape[::-1],
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            transformed_2d[tmp_registered > 0] = label

        transformed_3d[k, :, :] = transformed_2d

    return transformed_3d


def register_intensity_images(
        img_path_fixed, img_path_moving, maxCount, epsilon):
    ''' Register the average intensity images of the two ophys
        sessions using affine transformation'''

    # read average intensity images
    img_fixed = np.array(im.open(img_path_fixed))
    img_moving = np.array(im.open(img_path_moving))

    # Define termination criteria
    criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            maxCount,
            epsilon)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        ccval, tform = cv2.findTransformECC(
                img_fixed,
                img_moving,
                np.eye(2, 3, dtype=np.float32),
                cv2.MOTION_AFFINE,
                criteria)
    except cv2.error:
        logger.error("failed to align images {} and {}".format(
            img_path_fixed,
            img_path_moving))
        raise

    img_moving_warped = cv2.warpAffine(
            img_moving,
            tform,
            img_fixed.shape[::-1],
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return tform, img_moving_warped


def imshow3d(img3d, display_row_num, display_col_num):
    '''Display 3d images.'''

    dep = np.shape(img3d)[0]
    if display_row_num * display_col_num < dep:
        logger.error("The number of row to display times the "
                     "number of columns is less than the depth "
                     "of the image. Fail to display.")
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

    img = sitk.ReadImage(filename.encode('utf-8'))
    dim = img.GetDimension()

    if dim not in [2, 3]:
        raise ValueError("Error in read_tiff_3d() Image "
                         "dimension must be 2 or 3.")

    img3d = sitk.GetArrayFromImage(img).astype('int')
    if dim == 2:
        img3d = np.expand_dims(img3d, axis=0)

    return img3d


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


class PairwiseMatching(ArgSchemaParser):
    default_schema = PairwiseMatchingSchema
    ''' Class for pairwise ophys matching.

        Images are registered using intensity correlation.
        Best matches between cells are determined by bipartite graph matching.
    '''

    def run(self):
        ''' Pairwise matching of ophys-ophys sessions. '''

        filename_output_fixed_matched_img = ''
        filename_output_moving_matched_img = ''

        pre_fixed = re.findall(
                self.args['id_pattern'],
                self.args['filename_intensity_fixed'])[0]
        pre_moving = re.findall(
                self.args['id_pattern'],
                self.args['filename_intensity_moving'])[0]

        filename_matching_table = pre_moving + '_to_' + pre_fixed

        # register the average intensity images
        tform, moving_warped = register_intensity_images(
                self.args['filename_intensity_fixed'],
                self.args['filename_intensity_moving'],
                self.args['registration_iterations'],
                self.args['registration_precision'])

        if self.args['save_registered_image'] == 1:
            filename = os.path.join(
                self.args['output_directory'],
                'register_%s_to_%s.tif' % (pre_moving, pre_fixed)
                ).encode('utf-8')
            sitk_img_moving_registered = sitk.GetImageFromArray(
                moving_warped.astype(np.uint8))
            print(filename)
            print(sitk_img_moving_registered.GetSize())
            sitk.WriteImage(sitk_img_moving_registered, filename)

        # relabel the masks and write to disk
        self.segmask_fixed_3d = relabel(
                read_tiff_3d(self.args['filename_segmask_fixed']))
        self.segmask_moving_3d = relabel(
                read_tiff_3d(self.args['filename_segmask_moving']))

        filename_segmask_fixed_relabel = os.path.join(
            self.args['output_directory'],
            pre_fixed +
            '_maxInt_masks_relabel.tif').encode('utf-8')

        sitk_segmask_fixed = sitk.GetImageFromArray(
            self.segmask_fixed_3d.astype(np.uint16))
        sitk.WriteImage(sitk_segmask_fixed, filename_segmask_fixed_relabel)

        filename_segmask_moving_relabel = os.path.join(
            self.args['output_directory'],
            pre_moving +
            '_maxInt_masks_relabel.tif').encode('utf-8')

        sitk_segmask_moving = sitk.GetImageFromArray(
            self.segmask_moving_3d.astype(np.uint16))
        sitk.WriteImage(sitk_segmask_moving, filename_segmask_moving_relabel)

        # transform moving segmentation masks
        # compute self.segmask_moving_3d_registered
        self.segmask_moving_3d_registered = transform_masks(
                self.segmask_moving_3d,
                self.segmask_fixed_3d.shape[1:],
                tform)

        # matching cells
        tmp_filenames = dict()
        tmp_filenames['fixed_matched_img'] = \
            filename_output_fixed_matched_img
        tmp_filenames['moving_matched_img'] = \
            filename_output_moving_matched_img
        tmp_filenames['matching_table'] = filename_matching_table

        [matching_ratio_fixed, matching_ratio_moving, weight_matrix] = \
            self.cell_matching(tmp_filenames)

        matching_pairs = dict()
        matching_pairs['res'] = self.matching_table
        matching_pairs['mr_i'] = matching_ratio_fixed
        matching_pairs['mr_j'] = matching_ratio_moving
        matching_pairs['segmask_i'] = self.segmask_fixed_3d
        matching_pairs['segmask_j'] = self.segmask_moving_3d_registered
        matching_pairs['weight_matrix'] = weight_matrix
        matching_pairs['moving'] = self.args['filename_intensity_moving']
        matching_pairs['fixed'] = self.args['filename_intensity_fixed']

        # if affine only, output full 3x3
        if tform.shape == (2, 3):
            tform = np.vstack((tform, [0, 0, 1]))

        matching_pairs['transform'] = np.round(tform, 6).tolist()

        return matching_pairs

    def write_matching_images(self, para):
        ''' Write matching results into images. The same
            label codes the same cell.'''

        if (
                (len(para['filename_fixed']) > 0) &
                (len(para['filename_moving']) > 0)):

            para['filename_fixed'] = os.path.join(
                self.args['output_directory'], self.args['filename_fixed'])
            para['filename_moving'] = os.path.join(
                self.args['output_directory'], self.args['filename_moving'])

            segmask_moving_3d_matching = np.zeros(
                    self.segmask_moving_3d_registered.shape)
            segmask_fixed_3d_matching = np.zeros(self.segmask_fixed_3d.shape)

            for i in range(self.matching_table.shape[0]):
                if (
                        self.matching_table[i, 0] > 0 and
                        self.matching_table[i, 1] > 0):

                    segmask_fixed_3d_matching[
                            self.segmask_fixed_3d ==
                            para['labels_fixed'][
                                self.matching_table[i, 0] - 1]] = \
                                    self.matching_table[i, 0]
                    segmask_moving_3d_matching[
                            self.segmask_moving_3d_registered ==
                            para['labels_moving'][
                                self.matching_table[i, 1] - 1]] = \
                        self.matching_table[i, 0]

            sitk_img1 = sitk.GetImageFromArray(
                    segmask_fixed_3d_matching.astype(np.uint16))
            sitk.WriteImage(sitk_img1, para['filename_fixed'])

            sitk_img2 = sitk.GetImageFromArray(
                    segmask_moving_3d_matching.astype(np.uint16))
            sitk.WriteImage(sitk_img2, para['filename_moving'])

    def gen_matching_table(self, para_matching):
        '''Generate self.matching_table using bipartite graph matching.'''

        if self.args['munkres_executable']:
            # C++
            tool_name_args = [self.args['munkres_executable'] + " " +
                              para_matching['filename_weightmatrix'] + " " +
                              str(para_matching['fixed_cellnum']) + " " +
                              str(para_matching['moving_cellnum']) + " " +
                              para_matching['filename_tmpmatching']]

            run_bipartite_matching(tool_name_args)

            # load matching result produced by bp_matching
            matching_pair = np.loadtxt(
                    para_matching['filename_tmpmatching'],
                    delimiter=' ').astype(int)

        else:
            # scipy
            matching_pair = np.transpose(
                    np.array(
                        scipy.optimize.linear_sum_assignment(
                            np.loadtxt(
                                para_matching['filename_weightmatrix']))))

        # remove pairs that do not satisfy distance condition
        # matching_num is the smaller value of the number of regions in fixed
        # or moving masks
        num_matched = 0

        # determine the number of rows in matching_table
        # include all cells in fixed image
        count = para_matching['fixed_cellnum']

        for i in range(para_matching['moving_cellnum']):
            ind = np.argwhere(matching_pair[:, 1] == i)
            if ind.size == 0:
                # add those moving cells that are not in matching_pair
                count = count + 1
            elif para_matching['edge_fea']['dist'][
                    int(matching_pair[ind[0][0], 0]),
                    int(matching_pair[ind[0][0], 1])] >= \
                    self.args['maximum_distance']:
                # revent cells too far from matching even
                # if matching_pair match them
                count = count + 1

        self.matching_table = np.zeros((count, 5))

        # add all cells in fixed image
        for i in range(para_matching['fixed_cellnum']):
            ind = np.argwhere(matching_pair[:, 0] == i)
            if ind.size != 0:
                if para_matching['edge_fea']['dist'][
                        int(matching_pair[ind[0], 0][0]),
                        int(matching_pair[ind[0], 1][0])] < \
                        self.args['maximum_distance']:
                    # label index starts from 0, but label value starts from 1,
                    self.matching_table[i, 0] = matching_pair[ind, 0] + 1
                    self.matching_table[i, 1] = matching_pair[ind, 1] + 1
                    self.matching_table[i, 2] = \
                        para_matching['edge_fea']['dist'][
                                int(matching_pair[ind[0], 0][0]),
                                int(matching_pair[ind[0], 1][0])]
                    self.matching_table[i, 3] = \
                        para_matching['edge_fea']['overlap'][
                                int(matching_pair[ind[0], 0][0]),
                                int(matching_pair[ind[0], 1][0])]
                    self.matching_table[i, 4] = \
                        para_matching['edge_fea']['weight_matrix'][
                                int(matching_pair[ind[0], 0][0]),
                                int(matching_pair[ind[0], 1][0])]
                    num_matched = num_matched + 1
                else:
                    # does not allow matching
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
            # notice self.matching_table, not matching_pair
            ind = np.argwhere(self.matching_table[:, 1] == i + 1)
            if ind.size == 0:
                # label index starts from 0, but label value starts from 1,
                self.matching_table[count, 0] = -1
                self.matching_table[count, 1] = i + 1
                self.matching_table[count, 2] = -1
                self.matching_table[count, 3] = -1
                self.matching_table[count, 4] = -1
                count = count + 1

        return num_matched

    def cell_matching(self, tmp_filenames):
        ''' Function that matches cells. '''

        # compute features
        filename_weightmatrix = os.path.join(
            self.args['output_directory'], 'weightmatrix.txt')
        fixed_fea, moving_fea, distance, overlap, cost_matrix = \
            calculate_cost_matrix(
                    self.segmask_fixed_3d,
                    self.segmask_moving_3d_registered,
                    self.args['maximum_distance'])

        np.savetxt(
            filename_weightmatrix,
            cost_matrix,
            delimiter=' ')

        # Generate matching result by calling bipartite graph matching in c++
        filename_tmpmatching = os.path.join(
            self.args['output_directory'], 'matching_result_temporary.txt')

        para_matching = dict()

        para_matching['filename_weightmatrix'] = filename_weightmatrix
        para_matching['filename_tmpmatching'] = filename_tmpmatching
        para_matching['fixed_cellnum'] = fixed_fea['labels'].size
        para_matching['moving_cellnum'] = moving_fea['labels'].size
        para_matching['edge_fea'] = {
                'dist': distance,
                'overlap': overlap,
                'weight_matrix': cost_matrix}

        num_matched = self.gen_matching_table(para_matching)

        # write matching images, matched cells are given the same label
        # unmatched cells are not labeled in the image
        para_w = dict()
        para_w['filename_fixed'] = tmp_filenames['fixed_matched_img']
        para_w['filename_moving'] = tmp_filenames['moving_matched_img']
        para_w['labels_fixed'] = fixed_fea['labels']
        para_w['labels_moving'] = moving_fea['labels']

        self.write_matching_images(para_w)

        # write matching table
        if len(tmp_filenames['matching_table']) > 0:
            filename_matching_table = os.path.join(
                self.args['output_directory'], tmp_filenames['matching_table'])
            np.savetxt(
                filename_matching_table,
                self.matching_table,
                delimiter=' ',
                fmt="%d %d %7.4e %7.4e %7.4e")

        matching_ratio_fixed = np.float(num_matched) / fixed_fea['labels'].size
        matching_ratio_moving = \
            np.float(num_matched) / moving_fea['labels'].size

        # remove temporary files
        remove_tmp_files(filename_weightmatrix)
        remove_tmp_files(filename_tmpmatching)

        return (
                matching_ratio_fixed,
                matching_ratio_moving,
                cost_matrix)


if __name__ == "__main__":
    pmod = PairwiseMatching()
    pmod.run()
