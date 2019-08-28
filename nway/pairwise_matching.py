# -*- coding: utf-8 -*-
"""

Pair-wise matching called by nway matching main function

Copyright (c) Allen Institute for Brain Science
"""
import os
import logging
import subprocess
import numpy as np
import json
from PIL import Image as im
import SimpleITK as sitk
import cv2
import scipy.spatial
import scipy.optimize
from argschema import ArgSchemaParser
import tempfile

from nway.schemas import PairwiseMatchingSchema
import nway.utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def gen_assignment_pairs(cost_matrix, cpp_executable=None):
    '''Generate self.matching_table using bipartite graph matching.'''

    cost_matrix_array = np.array(cost_matrix)

    if cpp_executable:
        # C++
        infile = tempfile.NamedTemporaryFile(delete=False).name
        outfile = tempfile.NamedTemporaryFile(delete=False).name
        np.savetxt(infile, cost_matrix_array, delimiter=' ')
        cpp_args = [
                cpp_executable,
                infile,
                str(cost_matrix_array.shape[0]),
                str(cost_matrix_array.shape[1]),
                outfile]
        subprocess.check_call(cpp_args)
        assigned_pairs = np.loadtxt(
                outfile, delimiter=' ').astype('int')
        [os.remove(i) for i in [infile, outfile] if os.path.isfile(i)]

    else:
        # scipy
        assigned_pairs = np.transpose(
                np.array(
                    scipy.optimize.linear_sum_assignment(
                        cost_matrix_array)))

    row_lab = cost_matrix.index.tolist()
    col_lab = cost_matrix.columns.tolist()
    pairlabels = [[row_lab[pair[0]], col_lab[pair[1]]] for pair in assigned_pairs]

    return pairlabels


def gen_matching_table(
        cost_matrix, assigned_pairs, distance, iou, max_distance):
    '''Generate self.matching_table using bipartite graph matching.'''

    num_matched = 0
    matching_table = []
    jmatched = []
    for ap in assigned_pairs:
        if distance[ap[1]][ap[0]] < max_distance:
            matching_table.append([
                ap[0],
                ap[1],
                distance[ap[1]][ap[0]],
                iou[ap[1]][ap[0]],
                cost_matrix[ap[1]][ap[0]]])
            jmatched.append(ap[1])
        else:
            matching_table.append([ap[0], -1, -1, -1, -1])
    for j in cost_matrix.columns.tolist():
        if j not in jmatched:
            matching_table.append([-1, ap[1], -1, -1, -1])

    #for i in range(distance.shape[0]):
    #    matching_table.append([i + 1, -1, -1, -1, -1])
    #    if i in assigned_pairs:
    #        j = assigned_pairs[i]
    #        if distance[i, j] < max_distance:
    #            matching_table[-1] = ([
    #                i + 1,
    #                j + 1,
    #                distance[i, j],
    #                iou[i, j],
    #                cost_matrix[i, j]])
    #            num_matched = num_matched + 1
    #            jmatched.append(j)
    #for j in range(distance.shape[1]):
    #    if j not in jmatched:
    #        matching_table.append([-1, j + 1, -1, -1, -1])

    return np.array(matching_table)


def save_matching_table(table, output_directory, basename):
    np.savetxt(
            os.path.join(output_directory, basename),
            table,
            delimiter=' ',
            fmt="%d %d %7.4e %7.4e %7.4e")


def region_properties(mask, mdict):
    """Characterize each label region in a mask

    Parameters
    ----------
    mask : :class:`numpy.ndarray`
        2D mask image (row x column) or
        3D mask image (n x row x column) (untested)

    Returns
    -------
    prop : dict
        'centers' : :class:`numpy.ndarray`
            (N x 2) Cartesian coordinates of region
            centers-of-mass
        'pixels' : set(tuples)
            (row, column) tuples of pixels for each label

    """
    if len(mask.shape) not in [2, 3]:
        raise ValueError("region_properties needs a 2D or 3D image")

    prop = {}
    n = mask.max()
    prop['centers'] = np.zeros((n, 2))
    prop['pixels'] = [[]] * n
    prop['labels'] = [[]] * n
    for i in np.arange(n):
        label_locs = np.argwhere(mask == i + 1)
        prop['labels'][i] = mdict[str(i + 1)]
        if len(label_locs) > 0:
            # NOTE: the rounding/int-cast here should not be here
            # leaving it in place for now to preserve legacy
            prop['centers'][i] = np.round(
                    label_locs.mean(axis=0)).astype('int')[[-1, -2]]
        prop['pixels'][i] = set([tuple(x) for x in label_locs])
    return prop


def calculate_distance_and_iou(mask1, mask2, dict1, dict2):
    """Compute distance and intersection-over-union
    for the labels in two masks

    Parameters
    ----------
    mask1 : :class:`numpy.ndarray`
        2D mask image (row x column) or
        3D mask image (n x row x column) (untested)
    mask2 : :class:`numpy.ndarray`
        2D mask image (row x column) or
        3D mask image (n x row x column) (untested)

    Returns
    -------
    distance : :class:`numpy.ndarray`
        n_unique_labels1 x n_unique_labels2,
        Euclidean distance of centers
    iou : :class:`numpy.ndarray`
        n_unique_labels1 x n_unique_labels2,
        intersection-over-union pixel area ratios

    """
    prop1 = region_properties(mask1, dict1)
    prop2 = region_properties(mask2, dict2)

    # pair-wise distances between cell centers
    distance = scipy.spatial.distance.cdist(
            prop1['centers'], prop2['centers'], 'euclidean')

    # intersection-over-union
    iou = np.zeros_like(distance)
    for i, ipix in enumerate(prop1['pixels']):
        for j, jpix in enumerate(prop2['pixels']):
            iou[i, j] = \
                    len(ipix.intersection(jpix)) / len(ipix.union(jpix))

    distance = utils.frame_from_array(
        distance, prop1['labels'], prop2['labels'])
    iou = utils.frame_from_array(
        iou, prop1['labels'], prop2['labels'])

    return distance, iou


def calculate_cost_matrix(distance, iou, maximum_distance):
    """Combine distance and iou into a cost

    Parameters
    ----------
    distance : :class:`numpy.ndarray`
        N x M array of label center distances
    iou : :class:`numpy.ndarray`
        N x M array of label ious
    maximum_distance : Int
        threshold beyond which a large cost
        will be incurred

    Returns
    -------
    cost_matrix : :class:`numpy.ndarray`
        N x M calculated cost matrix

    """
    norm_dist = np.array(distance) / maximum_distance
    norm_dist[norm_dist > 1] = 999.0
    cost_matrix = norm_dist + (1.0 - np.array(iou))
    cost_matrix = utils.frame_from_array(
            cost_matrix, distance.index.tolist(), distance.columns.tolist())
    return cost_matrix


def cell_matching(mask1, mask2, dict1, dict2, max_dist, munkres_exe):
    ''' Function that matches cells. '''

    distance, iou = calculate_distance_and_iou(mask1, mask2, dict1, dict2)

    cost_matrix = calculate_cost_matrix(distance, iou, max_dist)

    assigned_pairs = gen_assignment_pairs(cost_matrix, munkres_exe)

    matching_table = gen_matching_table(
            cost_matrix,
            assigned_pairs,
            distance,
            iou,
            max_dist)

    return matching_table, cost_matrix


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


cvmotion = {
        "MOTION_TRANSLATION": cv2.MOTION_TRANSLATION,
        "MOTION_EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
        "MOTION_AFFINE": cv2.MOTION_AFFINE,
        "MOTION_HOMOGRAPHY": cv2.MOTION_HOMOGRAPHY}


def register_intensity_images(
        img_path_fixed, img_path_moving, maxCount, epsilon,
        save_image, output_directory, basename, motiontype):
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
                cvmotion[motiontype],
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

    if save_image:
        sitk.WriteImage(
                sitk.GetImageFromArray(img_moving_warped.astype(np.uint8)),
                os.path.join(output_directory, basename).encode('utf-8'))

    return tform, img_moving_warped


class PairwiseMatching(ArgSchemaParser):
    default_schema = PairwiseMatchingSchema
    ''' Class for pairwise ophys matching.

        Images are registered using intensity correlation.
        Best matches between cells are determined by bipartite graph matching.
    '''

    def run(self):
        ''' Pairwise matching of ophys-ophys sessions. '''

        self.logger.info('Matching %d to %d ...' % (
            self.args['fixed']['id'],
            self.args['moving']['id']))

        # identifying strings for filenames
        fixed_strid = '%s' % self.args['fixed']['id']
        moving_strid = '%s' % self.args['moving']['id']

        # register the average intensity images
        self.tform, moving_warped = register_intensity_images(
                self.args['fixed'][
                    'ophys_average_intensity_projection_image'],
                self.args['moving'][
                    'ophys_average_intensity_projection_image'],
                self.args['registration_iterations'],
                self.args['registration_precision'],
                self.args['save_registered_image'],
                self.args['output_directory'],
                'register_%s_to_%s.tif' % (moving_strid, fixed_strid),
                self.args['motionType'])

        segmask_fixed_3d = utils.read_tiff_3d(
                self.args['fixed']['nice_mask_path'])
        segmask_moving_3d = utils.read_tiff_3d(
                self.args['moving']['nice_mask_path'])

        # transform moving segmentation masks
        # compute self.segmask_moving_3d_registered
        segmask_moving_3d_registered = transform_masks(
                segmask_moving_3d,
                segmask_fixed_3d.shape[1:],
                self.tform)

        with open(self.args['fixed']['nice_dict_path'], 'r') as f:
            fixed_dict = json.load(f)
        with open(self.args['moving']['nice_dict_path'], 'r') as f:
            moving_dict = json.load(f)

        # matching cells
        self.matching_table, self.cost_matrix = cell_matching(
                    segmask_fixed_3d,
                    segmask_moving_3d_registered,
                    fixed_dict,
                    moving_dict,
                    self.args['maximum_distance'],
                    self.args['munkres_executable'])

        cname = os.path.join(
                self.args['output_directory'],
                moving_strid + '_to_' + fixed_strid + '_cost.npy')
        np.save(cname, self.cost_matrix)

        if self.args['save_pairwise_tables']:
            save_matching_table(
                    self.matching_table,
                    self.args['output_directory'],
                    moving_strid + '_to_' + fixed_strid)

        # if affine only, output full 3x3
        if self.tform.shape == (2, 3):
            self.tform = np.vstack((self.tform, [0, 0, 1]))

        return


if __name__ == "__main__":
    pmod = PairwiseMatching()
    pmod.run()
