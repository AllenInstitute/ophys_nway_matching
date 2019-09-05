import SimpleITK as sitk
import numpy as np
from skimage import measure as ms
import pandas as pd


def frame_from_array(a, rows, columns):
    df = pd.DataFrame(
        a,
        columns=columns,
        index=rows)
    return df


def relabel(maskimg_3d):
    ''' Relabel mask image to make labels continous and unique'''

    remask = np.copy(maskimg_3d)

    num_images = np.shape(remask)[0]
    labeloffset = 0

    for k in range(0, num_images):
        labelimg = ms.label((remask[k, :, :] > 0))
        labelimg[labelimg > 0] = labelimg[labelimg > 0] + labeloffset
        remask[k, :, :] = labelimg
        labeloffset = np.amax(labelimg)

    return remask


def row_col_from_roi(roi):
    x0 = roi["x"]
    y0 = roi["y"]
    w = roi["width"]
    h = roi["height"]
    coord = np.mgrid[y0:(y0 + h):1, x0:(x0 + w):1]
    mask = np.array(roi['mask_matrix'])
    masked = np.array([coord[i][mask]
                       for i in range(coord.shape[0])]).reshape(2, -1).T
    return masked


def labeled_mask_from_experiment(exp, legacy=True):
    im = sitk.GetImageFromArray(
            read_tiff_3d(
                exp['ophys_average_intensity_projection_image']))
    imsz = im.GetSize()
    zs = np.unique([r['z'] for r in exp['cell_rois']])
    mask = np.zeros((zs.size, imsz[1], imsz[0])).astype('uint32')
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

    # NOTE: above labeling should work on its own
    # but, some legacy order-dependence with the
    # Hungarian method requires the following ordering
    # Hungarian method is not recommended
    if legacy:
        relabeled = relabel(relabeled)
        rdict = {}
        for k in np.unique(relabeled)[1:]:
            ind = np.nonzero(relabeled == k)
            mi = mask[ind].flatten()
            assert np.unique(mi).size == 1
            rdict[str(k)] = int(mi[0])

    return relabeled, rdict


def calc_first_order_properties(M, force_shear='x'):
    """calculate scale shear and rotation from a 2x2 matrix
       could be M[0:2, 0:2] from an AffineModel or params[:, 1:3]
       from a Polynomial2DTransform
       copied from
    https://github.com/fcollman/render-python/blob/master/renderapi/transform/leaf/common.py
    Parameters
    ----------
    M : numpy array
        2x2
    force_shear : str
        'x' or 'y'
    Returns
    -------
    sx : float
        scale in x direction
    sy : float
        scale in y direction
    cx : float
        shear in x direction
    cy : float
        shear in y direction
    theta : float
        rotation angle in radians
    """
    if force_shear == 'x':
        sy = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
        theta = np.arctan2(M[1, 0], M[1, 1])
        rc = np.cos(theta)
        rs = np.sin(theta)
        sx = rc * M[0, 0] - rs * M[0, 1]
        if rs != 0:
            cx = (M[0, 0] - sx*rc) / (sx * rs)
        else:
            cx = (M[0, 1] - sx*rs) / (sx * rc)
        cy = 0.0
    elif force_shear == 'y':
        # shear in y direction
        sx = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        theta = np.arctan2(-M[0, 1], M[0, 0])
        rc = np.cos(theta)
        rs = np.sin(theta)
        sy = rs * M[1, 0] + rc * M[1, 1]
        if rs != 0:
            cy = (M[1, 1] - sy * rc) / (-sy * rs)
        else:
            cy = (M[1, 0] - sy * rs) / (sy * rc)
        cx = 0.0
    else:
        raise ValueError("%s not a valid option for force_shear."
                         " should be 'x' or 'y'")
    # room for other cases, for example cx = cy

    summary = {
            "scale x": sx,
            "scale y": sy,
            "shear x": cx,
            "shear y": cy,
            "rotation": theta,
            "translation x": M[0, 2],
            "translation y": M[1, 2]}
    if not np.all(M[2] == np.array([0.0, 0.0, 1.0])):
        summary['warning'] = "projective transform"

    return summary


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
