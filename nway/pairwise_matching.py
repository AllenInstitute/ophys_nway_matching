import os
import logging
import subprocess
import numpy as np
import pandas as pd
import json
from PIL import Image as im
import SimpleITK as sitk
import cv2
import scipy.spatial
import scipy.optimize
import networkx as nx
import re
import itertools
from argschema import ArgSchemaParser
import tempfile

from nway.schemas import PairwiseMatchingSchema, PairwiseOutputSchema
import nway.utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def gen_assignment_pairs(cost_matrix, solver, hungarian_exe):
    """generate pairs via an assignment problem solver

    Parameters
    ----------
    cost_matrix : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        N x M calculated cost matrix
    solver : str
       one of "Hungarian-cpp", "Hungarian" (scipy), or "Blossom"
    hungarian_exe : str
       absolute path to compiled Hungarian executable

    Returns
    -------
    pairlabels : list
        pairs of labels [label from row (index), label from column]
        that are the assigned matches

    """

    cost_matrix_array = np.array(cost_matrix)

    if solver == 'Hungarian-cpp':
        infile = tempfile.NamedTemporaryFile(delete=False).name
        outfile = tempfile.NamedTemporaryFile(delete=False).name
        np.savetxt(infile, cost_matrix_array, delimiter=' ')
        cpp_args = [
                hungarian_exe,
                infile,
                str(cost_matrix_array.shape[0]),
                str(cost_matrix_array.shape[1]),
                outfile]
        subprocess.check_call(cpp_args)
        assigned_pairs = np.loadtxt(
                outfile, delimiter=' ').astype('int')
        [os.remove(i) for i in [infile, outfile] if os.path.isfile(i)]

    elif solver == 'Hungarian':
        assigned_pairs = np.transpose(
                np.array(
                    scipy.optimize.linear_sum_assignment(
                        cost_matrix_array)))
    elif solver == 'Blossom':
        nrow, ncol = cost_matrix.shape
        i, j = np.mgrid[0:nrow:1, 0:ncol:1]
        # Blossom method is max weight, Hungarian is min weight
        weight = 1.0 / (1.0 + cost_matrix)
        ind = np.nonzero(weight > 0.002)
        etuples = [('r_%d' % ii, 'c_%d' % jj, {'weight': ww})
                   for ii, jj, ww in zip(
                       i[ind].flatten(),
                       j[ind].flatten(),
                       weight[ind].flatten())]
        G = nx.Graph()
        G.add_edges_from(etuples)
        k = nx.max_weight_matching(G)

        def get_pair_from_tuple(tup):
            rs = re.compile("r_(\d+)")
            cs = re.compile("c_(\d+)")
            x = " ".join(tup)
            pair = [
                    int(rs.search(x).groups()[0]),
                    int(cs.search(x).groups()[0])]
            return pair
        assigned_pairs = [get_pair_from_tuple(ik) for ik in k]

    row_lab = cost_matrix.index.tolist()
    col_lab = cost_matrix.columns.tolist()
    pairlabels = [[row_lab[pair[0]], col_lab[pair[1]]]
                  for pair in assigned_pairs]

    return pairlabels


def gen_matching_table(
        cost_matrix, assigned_pairs, distance, iou, max_distance):
    """generate rich summary of the matched pairs

    Parameters
    ----------
    cost_matrix : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        N x M calculated cost matrix
    assigned_pairs : list
        pairs of labels [label from row (index), label from column]
        that are the assigned matches
    distance : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        Euclidean distance of centers
    iou : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        intersection-over-union pixel area ratios
    maximum_distance : Int
        threshold beyond which a large cost
        will be incurred

    Returns
    -------
    matches : list of dict
        list of matches
        each match:
        {
          "distance": euclidean distance between centers
          "iou": intersection over union of mask pixels
          "fixed": fixed cell id
          "moving": moving cell id
          "cost": calculated cost
        }
    rejected : list of dict
        list of pairs within max_dist threshold which were not chosen
        same structure as matches

    """

    # candidates within max_distance, but not assigned
    rejected = []
    matches = []
    cols = cost_matrix.columns.tolist()
    rows = cost_matrix.index.tolist()
    for row, col in itertools.product(rows, cols):
        if distance[col][row] < max_distance:
            imatch = {
                    "fixed": row,
                    "moving": col,
                    'distance': distance[col][row],
                    'iou': iou[col][row],
                    'cost': cost_matrix[col][row]}
            if [row, col] in assigned_pairs:
                matches.append(imatch)
            else:
                rejected.append(imatch)

    return matches, rejected


def region_properties(mask, mdict):
    """Characterize each label region in a mask

    Parameters
    ----------
    mask : :class:`numpy.ndarray`
        3D mask image (n x row x column)

    Returns
    -------
    prop : dict
        'centers' : :class:`numpy.ndarray`
            (N x 2) Cartesian coordinates of region
            centers-of-mass
        'pixels' : set(tuples)
            (row, column) tuples of pixels for each label

    """
    prop = {}
    n = mask.max()
    prop['centers'] = np.zeros((n, 2))
    prop['pixels'] = [[]] * n
    prop['labels'] = [[]] * n
    for i in np.arange(n):
        label_locs = np.argwhere(mask == i + 1)
        prop['labels'][i] = mdict['mask_dict'][str(i + 1)]
        if len(label_locs) > 0:
            # NOTE: the rounding/int-cast here should not be here
            # leaving it in place for now to preserve legacy
            prop['centers'][i] = np.round(
                    label_locs.mean(axis=0)).astype('int')[[-1, -2]]
        prop['pixels'][i] = set([tuple(x) for x in label_locs])

    # sometimes a cell ROI gets transformed outside
    # add a null column for it
    for label in mdict['mask_dict'].values():
        if label not in prop['labels']:
            prop['labels'].append(label)
            prop['centers'] = np.append(
                    prop['centers'],
                    np.array([0, 0]).reshape(1, -1), axis=0)
            prop['pixels'].append(set())
    return prop


def calculate_distance_and_iou(mask1, mask2, dict1, dict2, legacy=False):
    """Compute distance and intersection-over-union
    for the labels in two masks

    Parameters
    ----------
    mask1 : :class:`numpy.ndarray`
        3D mask image (n x row x column)
    mask2 : :class:`numpy.ndarray`
        3D mask image (n x row x column)
    dict1 : dict
        mapping for mask1
        {
          "experiment": id of experiment
          "mask_path": absolute path to mask
          "mask_dict": {str(intensity): cellid}
        }
    dict2 : dict
        mapping for mask2
    legacy : bool
        whether to preserve legacy behavior

    Returns
    -------
    distance : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        Euclidean distance of centers
    iou : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
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
            intersection = len(ipix.intersection(jpix))
            union = len(ipix.union(jpix))
            if legacy:
                # NOTE
                # the orginal was written in python 2 and
                # the IOU calculation almost always gave 0 :(
                # as does this still
                iou[i, j] = intersection // union
            else:
                iou[i, j] = float(intersection) / union

    distance = pd.DataFrame(distance, prop1['labels'], prop2['labels'])
    iou = pd.DataFrame(iou, prop1['labels'], prop2['labels'])

    return distance, iou


def calculate_cost_matrix(distance, iou, maximum_distance):
    """Combine distance and iou into a cost

    Parameters
    ----------
    distance : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        Euclidean distance of centers
    iou : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        intersection-over-union pixel area ratios
    maximum_distance : Int
        threshold beyond which a large cost
        will be incurred

    Returns
    -------
    cost_matrix : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        N x M calculated cost matrix

    """
    norm_dist = np.array(distance) / maximum_distance
    norm_dist[norm_dist > 1] = 999.0
    cost_matrix = norm_dist + (1.0 - np.array(iou))
    cost_matrix = pd.DataFrame(
            cost_matrix, distance.index.tolist(), distance.columns.tolist())
    return cost_matrix


def cell_matching(
        mask1, mask2, dict1, dict2, max_dist, solver, hungarian_exe,
        legacy=False):
    """matching between 2 masks

    Parameters
    ----------
    mask1 : :class:`numpy.ndarray`
        3D mask image (n x row x column)
    mask2 : :class:`numpy.ndarray`
        3D mask image (n x row x column)
    dict1 : dict
        mapping for mask1
        {
          "experiment": id of experiment
          "mask_path": absolute path to mask
          "mask_dict": {str(intensity): cellid}
        }
    dict2 : dict
        mapping for mask2
    max_dist : int
        maximum distance threshold [pixels]. In the case of
        the Hungarian minimum weight algorithm, pairs beyond this
        threshold will receive a very large weight. In the case
        of the Blossom algorithm, pairs beyond this threshold will
        not be part of the graph.
    solver : str
       one of "Hungarian-cpp", "Hungarian" (scipy), or "Blossom"
    hungarian_exe : str
       absolute path to compiled Hungarian executable
    legacy : bool
        whether to preserve legacy behavior

    Returns
    -------
    cost_matrix : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        N x M calculated cost matrix
    matches : list of dict
        list of matches
        each match:
        {
          "distance": euclidean distance between centers
          "iou": intersection over union of mask pixels
          "fixed": fixed cell id
          "moving": moving cell id
          "cost": calculated cost
        }
    rejected : list of dict
        list of pairs within max_dist threshold which were not chosen
        same structure as matches

    """

    distance, iou = calculate_distance_and_iou(
        mask1, mask2, dict1, dict2, legacy=legacy)

    cost_matrix = calculate_cost_matrix(distance, iou, max_dist)

    assigned_pairs = gen_assignment_pairs(cost_matrix, solver, hungarian_exe)

    matches, rejected = gen_matching_table(
            cost_matrix,
            assigned_pairs,
            distance,
            iou,
            max_dist)

    return cost_matrix, matches, rejected


def transform_mask(moving, dst_shape, tform):
    """warp a mask according to a transform

    Parameters
    ----------
    moving : :class:`numpy.ndarray`
        nlayers x n x m int
    dst_shape : tuple
        (n', m') shape of destination layers
    tform : :class:`numpy.ndarry`
        3 x 3 transformation matrix

    Returns
    -------
    transformed_3d : :class:`numpy.ndarray`
        nlayer x n' x m' int

    """

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
            tmp_registered = cv2.warpPerspective(
                    tmp,
                    tform,
                    dst_shape[::-1],
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            transformed_2d[tmp_registered > 0] = label

        transformed_3d[k, :, :] = transformed_2d

    return transformed_3d


def register_intensity_images(
        img_path_fixed, img_path_moving, maxCount, epsilon, motion_type):
    """find the transform that registers two images

    Parameters
    ----------
    img_path_fixed : str
        path to fixed image
    img_path_moving : str
        path to moving image
    maxCount : int
        passed as maxCount to opencv termination criteria
    epsilon : float
        passed as epsilon to opencv termination criteria
    motion_type : str
        one of the 4 possible motion types for opencv findTransformECC

    Returns
    -------
    tform : :class:`numpy.ndarry`
        3 x 3 transformation matrix

    img_moving_warped : :class:`numpy.ndarray`
        warped moving image, same shape as fixed image, uint8

    """

    cvmotion = {
            "MOTION_TRANSLATION": cv2.MOTION_TRANSLATION,
            "MOTION_EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
            "MOTION_AFFINE": cv2.MOTION_AFFINE,
            "MOTION_HOMOGRAPHY": cv2.MOTION_HOMOGRAPHY}

    # read average intensity images
    img_fixed = np.array(im.open(img_path_fixed))
    img_moving = np.array(im.open(img_path_moving))

    # Define termination criteria
    criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            maxCount,
            epsilon)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    if motion_type == 'MOTION_HOMOGRAPHY':
        warp_matrix = np.eye(3, 3, dtype=np.float32)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        ccval, tform = cv2.findTransformECC(
                img_fixed,
                img_moving,
                warp_matrix,
                cvmotion[motion_type],
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
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP).astype(np.uint8)

    # if affine only, output full 3x3
    if tform.shape == (2, 3):
        tform = np.vstack((tform, [0, 0, 1]))

    return tform, img_moving_warped


class PairwiseMatching(ArgSchemaParser):
    default_schema = PairwiseMatchingSchema
    default_output_schema = PairwiseOutputSchema
    """Class for matching cells between optical physiology sessions
    """

    def run(self):
        """main function call for PairwiseMatching
        """
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
                self.args['motionType'])

        if self.args['save_registered_image']:
            imfname = os.path.join(
                self.args['output_directory'],
                'register_%s_to_%s.tif' % (moving_strid, fixed_strid))
            sitk.WriteImage(
                sitk.GetImageFromArray(moving_warped),
                imfname.encode('utf-8'))

        segmask_fixed_3d = utils.read_tiff_3d(
                self.args['fixed']['nice_mask_path'])
        segmask_moving_3d = utils.read_tiff_3d(
                self.args['moving']['nice_mask_path'])

        # transform moving segmentation mask
        segmask_moving_3d_registered = transform_mask(
                segmask_moving_3d,
                segmask_fixed_3d.shape[1:],
                self.tform)

        with open(self.args['fixed']['nice_dict_path'], 'r') as f:
            fixed_dict = json.load(f)
        with open(self.args['moving']['nice_dict_path'], 'r') as f:
            moving_dict = json.load(f)

        # matching cells
        self.cost_matrix, self.matches, self.rejected = cell_matching(
                    segmask_fixed_3d,
                    segmask_moving_3d_registered,
                    fixed_dict,
                    moving_dict,
                    self.args['maximum_distance'],
                    self.args['assignment_solver'],
                    self.args['hungarian_executable'],
                    legacy=self.args['legacy'])

        output_json = {
                'matches': self.matches,
                'rejected': self.rejected,
                'fixed_experiment': self.args['fixed']['id'],
                'moving_experiment': self.args['moving']['id'],
                'transform': {
                    "properties": utils.calc_first_order_properties(
                        self.tform),
                    'matrix': self.tform.tolist()
                    }
                }
        self.output(output_json, indent=2)

        return


if __name__ == "__main__":
    pmod = PairwiseMatching()
    pmod.run()
