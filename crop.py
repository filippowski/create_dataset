#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import resize
from skimage.io import imsave
import copy
import random
import warnings
warnings.filterwarnings("ignore")

from util import labels_array_to_list


# CREATE CROP
class Crop:

    def __init__(self, img, img_points, imgsize, cropsize, do_shft):
        self.img = img
        self.imgsize = imgsize
        self.cropsize = cropsize
        self.img_points = img_points
        self.scaled_img_points = None
        self.do_shft = do_shft # whether shift center?
        self.cntr_pt = 37 if len(img_points) == 119 else 35
        self.coef = 2.    # coefficient for scale crop delta
        self.shft = 10    # shift in pixels
        self.img_cropped = None
        self.x_offset    = None
        self.y_offset    = None
        self.new_size    = None

    def create_crop_transform_up(self, imgsize):
        translate_to_origin = np.identity(3)
        translate_to_origin[0, 2] = 1.
        translate_to_origin[1, 2] = 1.

        scale = imgsize * np.identity(3)

        transform = np.identity(3)
        transform = transform.dot(translate_to_origin)
        transform = transform.dot(scale)

        return transform

    def create_crop_transform_down(self, imgsize):
        translate_to_origin = np.identity(3)
        translate_to_origin[0, 2] = -float(imgsize)
        translate_to_origin[1, 2] = -float(imgsize)

        scale = np.identity(3) / float(imgsize)

        transform = np.identity(3)
        transform = transform.dot(translate_to_origin)
        transform = transform.dot(scale)

        return transform

    def get_center_and_delta(self, img_points):

        min_left_x  = int(np.amin([img_points[0][0],  img_points[1][0],  img_points[2][0],   img_points[3][0]], axis=0))
        max_right_x = int(np.amax([img_points[13][0], img_points[14][0], img_points[15][0],  img_points[16][0]], axis=0))
        min_top_y   = int(np.amin([img_points[17][1], img_points[18][1], img_points[19][1]], axis=0))
        max_bot_y   = int(np.amax([img_points[7][1],  img_points[8][1],  img_points[9][1]],  axis=0))

        delta = np.mean([img_points[self.cntr_pt][0] - min_left_x, max_right_x - img_points[self.cntr_pt][0]])
        x_cntr = min_left_x + delta
        y_cntr_1 = img_points[self.cntr_pt][1]
        y_cntr_2 = int(round(np.mean([min_top_y, max_bot_y])))
        y_cntr = int(round(np.mean([y_cntr_1, y_cntr_2])))

        return self.coef * delta, x_cntr, y_cntr


    def random_offset_cntr(self, delta, x_cntr, y_cntr):
        shft = self.cropsize / (2. * self.shft)
        shft = int(round(delta / shft))
        x_cntr_shifted = x_cntr + random.randint(-shft, shft)
        y_cntr_shifted = y_cntr + random.randint(-shft, shft)

        return x_cntr_shifted, y_cntr_shifted


    def gauss_offset_cntr(self, delta, x_cntr, y_cntr):
        sigma, mu = 3, 0
        sigma = self.cropsize / (2. * sigma)
        sigma = delta/sigma
        x_cntr_shifted = x_cntr + sigma * np.random.randn() + mu
        y_cntr_shifted = y_cntr + sigma * np.random.randn() + mu

        return x_cntr_shifted, y_cntr_shifted

    def rescale_pts(self):
        x = self.img_points[0::2]
        y = self.img_points[1::2]
        array = np.zeros([3, x.shape[0]])
        array[0] = x
        array[1] = y
        array[2] = 1.0
        fp = array.T

        t = self.create_crop_transform_up(0.5 * self.imgsize)
        fp = np.transpose(fp)
        fp = t.dot(fp)
        fp = np.transpose(fp)

        scaled_img_points = np.zeros(self.img_points.shape, dtype='float32')
        scaled_img_points[0::2] = fp.T[0]
        scaled_img_points[1::2] = fp.T[1]

        self.scaled_img_points = scaled_img_points

    def crop_head(self):

        # TO DO: yet scaled pts
        img_points = labels_array_to_list(self.img_points) if self.scaled_img_points is None else labels_array_to_list(self.scaled_img_points)

        img_cp = copy.copy(self.img)
        delta, x_cntr, y_cntr = self.get_center_and_delta(img_points)

        # random offset
        if self.do_shft:
            x_cntr, y_cntr = self.random_offset_cntr(delta, x_cntr, y_cntr)
            # gauss normal random offset
            #x_cntr, y_cntr = self.gauss_offset_cntr(delta, x_cntr, y_cntr)

        x_low = int(round(x_cntr - delta))
        x_high = int(round(x_cntr + delta))
        y_low = int(round(y_cntr - delta))
        y_high = int(round(y_cntr + delta))

        y_diff = y_high - y_low
        x_diff = x_high - x_low

        x_offset = x_low
        y_offset = y_low

        if x_low < 0 or y_low < 0 or x_high > self.img.shape[1] - 1 or y_high > self.img.shape[0] - 1:
            head_area = np.random.randint(255, size=(y_diff, x_diff, 3)).astype('uint8')  # uniform noise background
            x0, y0 = 0, 0
            x1, y1 = x_diff, y_diff
            if x_low < 0:
                x0 = abs(x_low)
                x_low = 0
            if y_low < 0:
                y0 = abs(y_low)
                y_low = 0
            if x_high > self.img.shape[1] - 1:
                x1 = x0 + self.img.shape[1] - x_low
                x_high = self.img.shape[1]
            else:
                x1 = x_diff
            if y_high > self.img.shape[0] - 1:
                y1 = y0 + self.img.shape[0] - y_low
                y_high = self.img.shape[0]
            else:
                y1 = y_diff
            print 'old bbox: ', x_low, x_high, y_low, y_high
            print 'new bbox: ', x0, x1, y0, y1
            head_area[y0:y1, x0:x1] = img_cp[y_low:y_high, x_low:x_high]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imgSizexSize = resize(head_area, (self.cropsize, self.cropsize, 3))
        else:
            print 'bbox: ', x_low, x_high, y_low, y_high
            head_area = img_cp[y_low:y_high, x_low:x_high]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imgSizexSize = resize(head_area, (self.cropsize, self.cropsize, 3))

        self.img_cropped = imgSizexSize
        self.x_offset    = x_offset
        self.y_offset    = y_offset
        self.new_size      = x_diff

    def save(self, path_to_crop_img):
        imsave(path_to_crop_img, self.img_cropped)