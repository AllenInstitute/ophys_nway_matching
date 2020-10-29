import os
import numpy as np
import pandas as pd
import json
import PIL.Image
import cv2
import scipy.spatial
import scipy.optimize
import networkx as nx
import itertools
from argschema import ArgSchemaParser

from nway.schemas import PairwiseMatchingSchema, PairwiseOutputSchema
from nway.meta_registration import MetaRegistration
import nway.utils as utils
import nway.image_processing_utils as imutils


class PairwiseException(Exception):
    pass


def gen_assignment_pairs(cost_matrix, solver):
    """generate pairs via an assignment problem solver

    Parameters
    ----------
    cost_matrix : :class:`pandas.DataFrame`
        index = labels1, columns = labels2
        N x M calculated cost matrix
    solver : str
       one of "Hungarian" (scipy), or "Blossom"

    Returns
    -------
    pairlabels : list
        pairs of labels [label from row (index), label from column]
        that are the assigned matches

    """

    cost_matrix_array = np.array(cost_matrix)

    if solver == 'Hungarian':
        assigned_pairs = np.transpose(
                np.array(
                    scipy.optimize.linear_sum_assignment(
                        cost_matrix_array)))
        row_lab = cost_matrix.index.tolist()
        col_lab = cost_matrix.columns.tolist()
        pairlabels = [[row_lab[pair[0]], col_lab[pair[1]]]
                      for pair in assigned_pairs]

    elif solver == 'Blossom':
        nrow, ncol = cost_matrix_array.shape
        # Blossom method is max weight, Hungarian is min weight
        weight = 1.0 / (1.0 + cost_matrix_array)
        ind = np.argwhere(weight > 0.002)
        etuples = []
        for i in ind:
            etuples.append((
                cost_matrix.index[i[0]],
                cost_matrix.columns[i[1]],
                {'weight': weight[i[0], i[1]]}))

        G = nx.Graph()
        G.add_edges_from(etuples)
        nxresults = nx.max_weight_matching(G)
        pairlabels = list([list(i) for i in nxresults])

        # sometimes networkx reverses the order
        for i in range(len(pairlabels)):
            if (
                    (pairlabels[i][0] in cost_matrix.index) &
                    (pairlabels[i][1] in cost_matrix.columns)):
                continue
            else:
                pairlabels[i] = pairlabels[i][::-1]

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
    mdict : dict
        mapping for mask
        {
          "experiment": id of experiment
          "mask_path": absolute path to mask
          "mask_dict": {str(intensity): cellid}
        }

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
            prop['centers'][i] = label_locs.mean(axis=0)[[-1, -2]]

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


def calculate_distance_and_iou(mask1, mask2, dict1, dict2):
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
    # NOTE for the Hungarian method, MAX_COST is high to be ignored
    # relative to the typical cost range [0, 2]. For the Blossom method
    # high costs will not populate the graph as possible edges
    MAX_COST = 999.0
    norm_dist[norm_dist > 1] = MAX_COST
    cost_matrix = norm_dist + (1.0 - np.array(iou))
    cost_matrix = pd.DataFrame(
            cost_matrix, distance.index.tolist(), distance.columns.tolist())
    return cost_matrix


def cell_matching(mask1, mask2, dict1, dict2, max_dist, solver):
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
    unmatched : dict
        list of cell IDs from fixed and moving experiments that were
        never considered for a pair

    """

    distance, iou = calculate_distance_and_iou(mask1, mask2, dict1, dict2)

    cost_matrix = calculate_cost_matrix(distance, iou, max_dist)

    assigned_pairs = gen_assignment_pairs(cost_matrix, solver)

    matches, rejected = gen_matching_table(
            cost_matrix,
            assigned_pairs,
            distance,
            iou,
            max_dist)

    # catalog what did not get paired at all
    paired_fixed_ids = [i['fixed'] for i in matches] + \
        [i['fixed'] for i in rejected]
    paired_moving_ids = [i['moving'] for i in matches] + \
        [i['moving'] for i in rejected]
    unmatched = {
            'fixed': [i for i in dict1['mask_dict'].values()
                      if i not in paired_fixed_ids],
            'moving': [i for i in dict2['mask_dict'].values()
                       if i not in paired_moving_ids]
            }

    return cost_matrix, matches, rejected, unmatched


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


class PairwiseMatching(ArgSchemaParser):
    default_schema = PairwiseMatchingSchema
    default_output_schema = PairwiseOutputSchema
    """Class for matching cells between optical physiology sessions
    """

    def run(self):
        """main function call for PairwiseMatching
        """
        self.logger.name = type(self).__name__
        logging_prefix = "Matching {} to {}".format(
                self.args['fixed']['id'],
                self.args['moving']['id'])
        self.logger.info(logging_prefix)

        # identifying strings for filenames
        fixed_strid = '%s' % self.args['fixed']['id']
        moving_strid = '%s' % self.args['moving']['id']

        # register the average intensity images
        fixp = self.args['fixed']['ophys_average_intensity_projection_image']
        movp = self.args['moving']['ophys_average_intensity_projection_image']
        with PIL.Image.open(fixp) as im:
            img_fixed = np.array(im)
        with PIL.Image.open(movp) as im:
            img_moving = np.array(im)

        meta_reg = MetaRegistration(
                maxCount=self.args['registration_iterations'],
                epsilon=self.args['registration_precision'],
                motion_type=self.args['motionType'],
                gaussFiltSize=self.args['gaussFiltSize'],
                CLAHE_grid=self.args['CLAHE_grid'],
                CLAHE_clip=self.args['CLAHE_clip'],
                edge_buffer=self.args['edge_buffer'],
                include_original=self.args['include_original'])
        meta_reg(img_moving, img_fixed)
        self.logger.info(f"{logging_prefix}: best registration was "
                         f"{meta_reg.best_candidate}")
        if meta_reg.best_candidate == ["Identity"]:
            raise PairwiseException(f"{logging_prefix}: no registration "
                                    "found")
        self.tform = meta_reg.best_matrix

        if self.args['save_registered_image']:
            moving_warped = imutils.warp_image(img_moving, self.tform,
                                               self.args['motionType'],
                                               img_fixed.shape)
            imfname = os.path.join(
                self.args['output_directory'],
                'register_%s_to_%s.tif' % (moving_strid, fixed_strid))
            with PIL.Image.fromarray(moving_warped) as im:
                im.save(imfname)

        # transform moving segmentation mask
        segmask_fixed_3d = utils.read_tiff_3d(
                self.args['fixed']['nice_mask_path'])
        segmask_moving_3d = utils.read_tiff_3d(
                self.args['moving']['nice_mask_path'])
        segmask_moving_3d_registered = transform_mask(
                segmask_moving_3d,
                segmask_fixed_3d.shape[1:],
                self.tform)

        with open(self.args['fixed']['nice_dict_path'], 'r') as f:
            fixed_dict = json.load(f)
        with open(self.args['moving']['nice_dict_path'], 'r') as f:
            moving_dict = json.load(f)

        # matching cells
        self.cost_matrix, self.matches, self.rejected, self.unmatched = \
            cell_matching(
                    segmask_fixed_3d,
                    segmask_moving_3d_registered,
                    fixed_dict,
                    moving_dict,
                    self.args['maximum_distance'],
                    self.args['assignment_solver'])

        # opencv likes float32, but json does not
        self.tform = self.tform.astype('float')
        output_json = {
                'unmatched': self.unmatched,
                'matches': self.matches,
                'rejected': self.rejected,
                'fixed_experiment': self.args['fixed']['id'],
                'moving_experiment': self.args['moving']['id'],
                'transform': {
                    "best_registration": meta_reg.best_candidate,
                    "properties": utils.calc_first_order_properties(
                        self.tform),
                    'matrix': self.tform.tolist()
                    }
                }
        self.output(output_json, indent=2)

        return


if __name__ == "__main__":  # pragma: no cover
    pmod = PairwiseMatching()
    pmod.run()
