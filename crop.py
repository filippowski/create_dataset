#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from skimage import io
import os
import copy
import random
import multiprocessing as mp
from  multiprocessing import cpu_count, Pool
import dlib
import fnmatch
import glob
import warnings
warnings.filterwarnings("ignore")

from util import labels_array_to_list, points_as_array, get_dist


# CREATE CROP
class Crop:

    def __init__(self, img, img_points, imgsize, crop_params):
        self.img = img
        self.img_cropped = None

        self.img_points = img_points
        self.scaled_img_points = None

        self.do_shft     = crop_params['do_shft']
        self.shft        = crop_params['shft']
        self.cntr_pt     = crop_params['cntr_pt']
        self.cntr_pt = 35 if len(img_points) == 113 else self.cntr_pt
        self.coef        = crop_params['coef']
        self.cropsize    = crop_params['imgSize']
        self.channel     = crop_params['channel']

        self.left_x      = crop_params['left_x']
        self.right_x     = crop_params['right_x']
        self.top_y       = crop_params['top_y']
        self.bot_y       = crop_params['bot_y']

        self.imgsize     = imgsize
        self.new_size    = None
        self.x_offset    = None
        self.y_offset    = None

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

        min_left_x  = int(np.amin([img_points[x][0] for x in self.left_x], axis=0))
        max_right_x = int(np.amax([img_points[x][0] for x in self.right_x], axis=0))
        min_top_y   = int(np.amin([img_points[x][1] for x in self.top_y], axis=0))
        max_bot_y   = int(np.amax([img_points[x][1] for x in self.bot_y], axis=0))

        # min_left_x  = int(np.amin([img_points[0][0],  img_points[1][0],  img_points[2][0],   img_points[3][0]], axis=0))
        # max_right_x = int(np.amax([img_points[13][0], img_points[14][0], img_points[15][0],  img_points[16][0]], axis=0))
        # min_top_y   = int(np.amin([img_points[17][1], img_points[18][1], img_points[19][1]], axis=0))
        # max_bot_y   = int(np.amax([img_points[7][1],  img_points[8][1],  img_points[9][1]],  axis=0))

        delta = np.mean([img_points[self.cntr_pt][0] - min_left_x, max_right_x - img_points[self.cntr_pt][0]])
        x_cntr = min_left_x + delta
        y_cntr_1 = img_points[self.cntr_pt][1]
        y_cntr_2 = int(round(np.mean([min_top_y, max_bot_y])))
        y_cntr = int(round(np.mean([y_cntr_1, y_cntr_2])))
        delta  = max_bot_y - y_cntr

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
            #print 'old bbox: ', x_low, x_high, y_low, y_high
            #print 'new bbox: ', x0, x1, y0, y1
            head_area[y0:y1, x0:x1] = img_cp[y_low:y_high, x_low:x_high]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imgSizexSize = resize(head_area, (self.cropsize, self.cropsize, 3))
        else:
            #print 'bbox: ', x_low, x_high, y_low, y_high
            head_area = img_cp[y_low:y_high, x_low:x_high]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imgSizexSize = resize(head_area, (self.cropsize, self.cropsize, 3))

        self.img_cropped = imgSizexSize
        self.x_offset    = x_offset
        self.y_offset    = y_offset
        self.new_size    = x_diff

    def save(self, path_to_crop_img):
        imsave(path_to_crop_img, self.img_cropped)


# CREATE CROP BY MEANS OF DLIB MODEL
class CropDLIB:

    def __init__(self, path_to_superdir, file_params, crop_params):

        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)

        self.bunch_fldname          = file_params['in']['alphas']['bunch_fldname']
        self.crop_endswith          = file_params['in']['dlib_model']['crop_endswith']
        self.imgs_ext               = file_params['in']['dlib_model']['imgs_ext']
        self.path_to_dlib_model     = file_params['in']['dlib_model']['path_to_model']
        assert os.path.exists(self.path_to_dlib_model), 'Path to DLIB model {} does not exist. Pls check path.'.format(
            self.path_to_dlib_model)

        self.crop_params   = crop_params

        self.nfolders      = 0
        self.nimgs         = 0
        self.queue         = self.get_queue()


    def get_queue(self):
        # Define folders queue
        queue = mp.Queue()

        # Put all paths to folders
        for root, subFolders, files in os.walk(self.path_to_superdir):
            for subFolder in subFolders:
                if subFolder[0:5] == self.bunch_fldname:
                    for root_, subFolders_, files_ in os.walk(os.path.join(root, subFolder)):
                        for subFolder_ in subFolders_:
                            path = os.path.join(root_, subFolder_)
                            file_count = len(fnmatch.filter(os.listdir(path), '*'+self.imgs_ext))
                            self.nfolders += 1
                            self.nimgs    += file_count
                            queue.put(os.path.join(root_, subFolder_))

        print 'In superdir {} are {} folders.'.format(self.path_to_superdir, self.nfolders)
        print 'In superdir {} are {} images.'.format(self.path_to_superdir, self.nimgs)

        return queue


    def run_multiprocessing_crop(self):

        # dlib detector and predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.path_to_dlib_model)

        print 'queue size: ', self.queue.qsize()

        # Setup a list of processes that we want to run
        func = self.crop_images_w_dlib_points
        args = (detector, predictor, self.queue)
        processes = [mp.Process(target=func, args=args) for x in range(self.queue.qsize())]

        nprocesses = len(processes)
        print 'cnt of processes: ', nprocesses
        nworkers = int(0.75 * mp.cpu_count())

        for i in range(int(nprocesses / nworkers) + 1):
            proc = processes[:nworkers]
            processes = processes[nworkers:]

            # Run processes
            for p in proc:
                p.start()

            # Exit the completed processes
            for p in proc:
                p.join()


    def crop_images_w_dlib_points(self, detector, predictor, queue):
        folder_path = queue.get()
        for f in glob.glob(os.path.join(folder_path, '*'+self.imgs_ext)):
            if not f.endswith(self.crop_endswith + self.imgs_ext):
                path_to_img = os.path.join(folder_path, f)
                img = io.imread(path_to_img)
                #print("Processing file: {}, ends crop: {}".format(f, f.endswith(self.crop_endswith + self.imgs_ext)))
                pts = self.get_dlib_points(detector, predictor, path_to_img)
                crop = Crop(img, pts, img.shape[0], self.crop_params)
                # if it is needed rescale pts
                # crop.rescale_pts()
                crop.crop_head()
                folder, fullfilename = os.path.split(f)
                filename, _ = os.path.splitext(fullfilename)
                new_path = os.path.join(folder, filename + self.crop_endswith + self.imgs_ext)
                crop.save(new_path)

    def get_dlib_points(self, detector, predictor, path_to_img):

        img = io.imread(path_to_img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        times = 0
        dets = detector(img, times)
        if len(dets) == 0:
            print("Number of faces detected: {}, times: {}, image: {}".format(len(dets), times, path_to_img))

        #if detector does not see any faces then upsanple image at most five times
        while (len(dets) < 1 and times < 5):
            times += 1
            dets = detector(img, times)
            print("Number of faces detected: {}, times: {}, image: {}".format(len(dets), times, path_to_img))

        max_d, max_dist = None, None
        for k, d in enumerate(dets):
            (max_d, max_dist) = (d, get_dist(d.left(), d.top(), d.right(), d.bottom())) if max_d is None else (max_d, max_dist)
            d_dist = get_dist(d.left(), d.top(), d.right(), d.bottom())
            (max_d, max_dist) = (d, d_dist) if d_dist > max_dist else (max_d, max_dist)

        # Get the landmarks/parts for the face in box d.
        points = predictor(img, max_d)
        points = points_as_array(points)
        # print points
        return points