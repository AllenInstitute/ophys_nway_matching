import numpy as np
from skimage.transform import AffineTransform
import PIL.Image
import os
import json


def create_nice_mask(experiment, output_directory):
    """create a labeled mask from the input json experiment and a translation
    dict to cell IDs, save them to disk, and add their paths to the experiment
    json.

    Parameters
    ----------
    experiment : dict
        output from segmentation
        'ophys_average_intensity_projection_image' : str(path)
        'max_int_mask_image' : str(path)
            (unused, this function creates unique label mask)
        'cell_rois' : dict
        'id' :  str
        'stimulus_name' : str/Null
    output_directory : str
        destination directory for writing masks and translation dicts

    Returns
    -------
    experiment : dict
        same as input with 2 additional key/value pairs
        'nice_mask_path' : str(path)
        'nice_dict_path' : str(path)

    """

    nice_mask, mask_dict = labeled_mask_from_experiment(experiment)

    mask_path = os.path.join(
            output_directory,
            "%d_nice_mask.tif" % experiment['id'])

    nice_mask = [PIL.Image.fromarray(ix) for ix in nice_mask]
    nice_mask[0].save(mask_path, save_all=True, append_images=nice_mask[1:])

    experiment['nice_mask_path'] = mask_path

    dict_path = os.path.join(
            output_directory,
            "%d_nice_dict.json" % experiment['id'])
    full_dict = {
            'experiment': experiment['id'],
            'mask_path': mask_path,
            'mask_dict': mask_dict
            }

    with open(dict_path, 'w') as f:
        json.dump(full_dict, f, indent=2)

    experiment['nice_dict_path'] = dict_path
    # marshmallow 3.0.0rc6 is less forgiving about extra keys around
    # so, pop out the unused extra keys here
    experiment.pop('stimulus_name')

    return experiment


def row_col_from_roi(roi):
    """from an roi dict, get a list of rows and columns
    that are True in the mask_matrix

    Parameters
    ----------
    roi : dict
        x: x (col) offset
        y: y (row) offset
        width: col width
        height: row height
        mask_matrix: nested boolean list h x w

    Returns
    -------
    masked : :class:`numpy.ndarry`
        N x 2 array of [row, col] pairs

    """
    x0 = roi["x"]
    y0 = roi["y"]
    w = roi["width"]
    h = roi["height"]
    coord = np.mgrid[y0:(y0 + h):1, x0:(x0 + w):1]
    mask = np.array(roi['mask_matrix'])
    masked = np.array([coord[i][mask]
                       for i in range(coord.shape[0])]).reshape(2, -1).T
    return masked


def labeled_mask_from_experiment(exp):
    """create a labeled mask and to-cell-id dictionary
    from the experiment json

    Parameters
    ----------
    exp : dict
        id : unique experiment id from LIMS
        ophys_average_intensity_projection_image : path to
        intensity image. used to establish lateral size
        of the mask
        cell_rois : list of dict
            x: x (col) offset
            y: y (row) offset
            width: col width
            height: row height
            mask_matrix: nested boolean list h x w
            id : unique cell id, uint32 from LIMS
            z : depth of roi. depths keep overlapping rois distinct

    Returns
    -------
    relabeled : :class:`numpy.ndarray`
        depth x n x m uint16 mask with new labels
    rdict : dict
        str(label in mask): cell ID

    """
    with PIL.Image.open(
            exp['ophys_average_intensity_projection_image']) as fim:
        im = np.array(fim)
    imsz = im.shape
    zs = np.unique([r['z'] for r in exp['cell_rois']])
    mask = np.zeros((zs.size, imsz[0], imsz[1])).astype('uint32')
    for roi in exp['cell_rois']:
        rc = row_col_from_roi(roi)
        mask[roi['z'], rc[:, 0], rc[:, 1]] = roi['id']

    x = np.unique(mask)
    relabeled = np.zeros_like(mask).astype('uint16')
    rdict = {}
    for i, ix in enumerate(x):
        ind = np.nonzero(mask == ix)
        relabeled[ind] = i
        rdict[str(i)] = int(ix)

    # do not list the background
    rdict.pop("0")

    return relabeled, rdict


def calc_first_order_properties(M):
    """human-readable affine properties of a transform matrix

    Parameters
    ----------
    M : :class:`numpy.ndarray`
       3 x 3 augmented affine or projective matrix

    Returns
    -------
    summary : dict
        human-readable dict of transform properties

    """

    summary = {}
    if not np.all(M[2] == np.array([0.0, 0.0, 1.0])):
        summary['warning'] = "affine properties of projective transform"

    tform = AffineTransform(M)
    summary['scale'] = tform.scale
    summary['shear'] = tform.shear
    summary['translation'] = tuple(tform.translation)
    summary['rotation'] = tform.rotation

    return summary


def read_tiff_3d(filename):
    """read in 3D tiff for mask

    Parameters
    ----------
    filename : str
        path to tiff file

    Returns
    -------
    img : :class:`numpy.ndarray`
        depth x n x m image
    """

    with PIL.Image.open(filename) as ims:
        sz = ims.size
        nframes = ims.n_frames
        im = np.zeros((nframes, sz[1], sz[0])).astype('uint16')
        for i, iim in enumerate(PIL.ImageSequence.Iterator(ims)):
            im[i] = np.array(iim)

    return im
