# -*- coding: utf-8 -*-
"""
# Compute image region properties
#
# Copyright (c) Allen Institute for Brain Science
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

plt.ion()



class RegionProperties(object):
    '''
    Class for computing region properties.

    label_img: labeled image
    img_width: width of the image
    img_heigth: height of the image
    img_depth: depth of the image (for 3D images)
    rgn_labels: label of each region
    rgn_num: the number of regions in the label image
    centers:2d array of centers of segmented regions: [[x1, y1], [x2, y2], [x3, y3], ...]
    areas: 1d array of region area.
    pixelidxlist: pixel index dictionary: [0: [[row1, col1], [row2, col2], ..., [row_k, col_k]], # region 1
                                      ...
                                      [[row1, col1], [row2, col2], ..., [row_p, col_p]]] # region n
    '''

    def __init__(self, img):
        self.img_height = 0
        self.img_width = 0
        self.img_depth = 0
        self.rgn_num = 0
        self.rgn_labels = []
        self.centers = []
        self.areas = []
        self.pixelidxlist = []
        self.label_img = np.copy(img)

    def get_size(self):
        ''' Get size of regions.'''

        if len(np.shape(self.label_img)) == 3:
            img_height = self.label_img.shape[2]
            img_width = self.label_img.shape[1]
            img_depth = self.label_img.shape[0]
            self.img_height = img_height
            self.img_width = img_width
            self.img_depth = img_depth
            return img_height, img_width, img_depth

        elif len(np.shape(self.label_img)) == 2:
            img_height = self.label_img.shape[0]
            img_width = self.label_img.shape[1]
            self.img_height = img_height
            self.img_width = img_width
            return img_height, img_width

        else:
            logger.error("Not 2D or 3D image, skip ...")
            return

    def get_labels(self):
        ''' Get all the labels of the regions in a labeled image.

        rgn_labels: label of each region,
        rgn_num: the number of regions
        '''

        # this requires label image to be indexed from 0 to N continously, remove background label 0
        rgn_labels = np.unique(range(1, np.amax(self.label_img) + 1))
        rgn_num = rgn_labels.size
        self.rgn_labels = rgn_labels
        self.rgn_num = rgn_num

        return rgn_labels, rgn_num

    def get_centers(self):
        '''Get center of mass of the regions.

        centers:2d array of centers of segmented regions: [[x1, y1], [x2, y2], [x3, y3], ...]
        '''

        self.get_labels()

        dim = len(np.shape(self.label_img))
        if dim == 2:
            centers = np.zeros((self.rgn_num, 2))
        elif dim == 3:
            centers = np.zeros((self.rgn_num, 3))
        else:
            logger.error("Not 2D or 3D image, skip ...")
            return

        for i in range(self.rgn_num):
            pixelidx = np.argwhere(self.label_img == self.rgn_labels[i])
            if len(pixelidx) > 0:
                centers[i] = np.round(np.mean(pixelidx.astype(np.float32), axis=0)).astype(np.int)
                if dim == 2:
                    centers[i] = centers[i][::-1]  # make the first x, the second y
                elif dim == 3:
                    centers[i][1:3] = centers[i][-1:-3:-1]
                else:
                    logger.error("Not 2D or 3D image, skip ...")
                    return

        self.centers = centers
        return centers

    def get_areas(self):
        '''Get region areas.

        areas: 1d array of region area.
        '''

        self.get_labels()  # does not mater if self.label_img is empty or not

        areas = np.zeros(self.rgn_num)

        for i in range(self.rgn_num):
            pixelidx = np.argwhere(self.label_img == self.rgn_labels[i])
            areas[i] = len(pixelidx)
        self.areas = areas

        return areas

    def get_pixel_idx_list(self):
        ''' Get the list of pixel indices. '''

        self.get_labels()
        pixelidxlist = dict()

        for i in range(self.rgn_num):
            pixelidx = np.argwhere(self.label_img == self.rgn_labels[i])
            pixelidxlist[self.rgn_labels[i]] = pixelidx

        self.pixelidxlist = pixelidxlist

        return pixelidxlist
