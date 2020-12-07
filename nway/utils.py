import numpy as np
from skimage.transform import AffineTransform
import PIL.Image
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Any, Tuple, Optional


def summarize_registration_success(pair_outputs):
    """checks which pairs succeeded in registratio. tries to keep
    dataframe small for logging by separating column/index names
    and returning an idmap also

    Parameters
    ----------
    pair_outputs: List[dict]
        at minimum, each entry is {
                                     'fixed_experiment': <int>,
                                     'moving_experiment': <int>,
                                     'transform': {
                                                    'best_registration': []
                                                  }
                                   }

    Returns
    -------
    df: pd.DataFrame
        boolean-valued dataframe, sorted both in rows and columns
        row/column names replaced with integers
    id_map: dict
        map of row/column names to experiment ids. can be used to reconstruct
        named rows and columns with:
        df = df.rename(index=id_map, columns=id_map)

    """
    df = pd.DataFrame()
    for pair_output in pair_outputs:
        with open(pair_output, "r") as f:
            j = json.load(f)
        success = 1
        if j['transform']['best_registration'] == ['Identity']:
            success = 0
        df.loc[j['fixed_experiment'], j['moving_experiment']] = success
        df.loc[j['moving_experiment'], j['fixed_experiment']] = success
    df = df.sort_index(0)
    df = df.sort_index(1)
    id_map = {v: i for i, v in enumerate(df.index)}
    for v in id_map.keys():
        df.loc[v, v] = 1
    df = df.rename(index=id_map, columns=id_map)
    df = df.astype(int)
    id_map = {v: k for k, v in id_map.items()}
    return df, id_map


def write_pair_images(output_path: Path, titles: List[str],
                      images: List[np.ndarray],
                      suptitle: Optional[str] = None) -> Path:
    """make a plot of images and write to disk

    Parameters
    ----------
    output_path: Path
        destination file
    titles: List[str]
        titles for the subplots
    images: List[np.ndarray]
        images for the subplots
    suptitle: str
        supertitle for the figure

    Return
    ------
    output_path: Path
        output path

    """
    if len(titles) != len(images):
        raise ValueError(f"length of titles {len(titles)} does not match "
                         f"length of images {len(images)}.")
    f, a = plt.subplots(1, len(images), clear=True, num=1, figsize=(12, 8))
    for ax, image, title in zip(a, images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
    if suptitle is not None:
        f.suptitle(suptitle)
    f.savefig(output_path)
    return output_path


def create_nice_mask(experiment, output_directory):
    """create a labeled mask from the input json experiment and a translation
    dict to cell IDs, save them to disk, and add their paths to the experiment
    json.

    Parameters
    ----------
    experiment : dict
        output from segmentation
        'ophys_average_intensity_projection_image' : str(path)
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


def layered_mask_from_rois(rois: List[Any],
                           shape: Tuple[int, int]) -> np.ndarray:
    """makes a (nz, *shape) dimensioned array where each layer ("z")
    has no overlapping ROIs

    Parameters
    ----------
    rois: list
        list of ROIs. The list is the same as "cell_rois" in
        ExperimentSchema
    shape: tuple
        (nrows, ncols) of the destination image

    Returns
    -------
    masks: numpy.ndarray
       dimensions (nz, nrows, ncols)

    """
    def new_layer():
        return np.zeros(shape, dtype=np.uint32)

    masks = [new_layer()]
    for roi in rois:
        tmp = new_layer()
        tmp[roi['y']:(roi['y'] + roi['height']),
            roi['x']:(roi['x'] + roi['width'])] = np.uint32(roi['mask_matrix'])
        tmp *= roi['id']
        bool_tmp = tmp != 0
        placed = False
        for imask in range(len(masks)):
            bool_mask = masks[imask] != 0
            if np.any(bool_tmp & bool_mask):
                # some overlap, move to next layer
                pass
            else:
                # no overlap, add to this layer
                masks[imask] += tmp
                placed = True
                break
        if not placed:
            masks.append(tmp)
    masks = np.array(masks)

    return masks


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
    # layered mask with uint32 ids as intensities
    mask = layered_mask_from_rois(exp['cell_rois'], im.shape)

    # relabel as ordered uint16 and a translating map
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
