#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import os
import csv
from scipy.ndimage import imread
from util import ensure_dir
from load import load_cls_labels, load_landmarks
from crop import Crop

# MERGE ALL CSVs IN ONE CSV
class Merge:

    def __init__(self, path_to_superdir, target_path, train_images_dir_name, file_params, task_names, imgSize, do_shft, mode):

        self.mode = mode
        assert self.mode in ['classification', 'landmarks', '3D'], \
            'Mode {} must be one from list {}. Pls check mode param.'.format(self.mode,'[classification, landmarks, 3D]')

        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), \
            'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)

        self.target_path = target_path
        assert os.path.exists(self.target_path), \
            'Path to path where must be saved general csv file and folder with resized images does not exist. Pls check path: {}.'.format(
            self.target_path)

        self.train_images_dir_name = train_images_dir_name

        self.task_names  = task_names
        self.do_shft     = do_shft
        self.imgsize     = imgSize
        self.cnt_mrg_img = 0

        if self.mode == 'landmarks':
                self.landmarks_filename     = file_params['landmarks']['csv_filename']
                self.landmarks_names        = file_params['landmarks']['names']
                self.landmarks_types        = file_params['landmarks']['types']
                self.landmarks_sep          = file_params['landmarks']['sep']

        if self.mode == 'classification':
                self.labels_filename        = file_params['labels']['csv_filename']
                self.labels_names           = file_params['labels']['names']
                self.labels_types           = file_params['labels']['types']
                self.labels_sep             = file_params['labels']['sep']

                self.microclasses_filename  = file_params['microclasses']['csv_filename']
                self.microclasses_names     = file_params['microclasses']['names']
                self.microclasses_types     = file_params['microclasses']['types']
                self.microclasses_sep       = file_params['microclasses']['sep']

                self.landmarks_filename     = file_params['landmarks']['csv_filename']
                self.landmarks_names        = file_params['landmarks']['names']
                self.landmarks_types        = file_params['landmarks']['types']
                self.landmarks_sep          = file_params['landmarks']['sep']

                self.path_to_labels         = os.path.join(self.path_to_superdir, self.labels_filename)
                self.path_to_microclasses   = os.path.join(self.path_to_superdir, self.microclasses_filename)
                self.path_to_landmarks      = os.path.join(self.path_to_superdir, self.landmarks_filename)


    def merge(self):
        if self.mode == 'classification':
            print ' * merge classification'
            self.merge_classification(self.path_to_superdir, self.target_path, self.train_images_dir_name, self.landmarks_filename, self.landmarks_sep, self.labels_filename, self.labels_sep)
        if self.mode == 'landmarks':
            print ' * merge landmarks'
            self.merge_landmarks(self.path_to_superdir, self.target_path, self.train_images_dir_name, self.landmarks_filename, self.landmarks_sep)
        if self.mode == '3D':
            print ' * merge 3D'
            #self.merge_func(self.path_to_superdir, self.target_path, self.train_images_dir_name)

    def merge_landmarks(self, path_to_superdir, target_path, train_images_dir_name, csv_filename, sep):
        '''
        path_to_superdir      - directory where are folders with images and csv-files
        target_path           - directory where will be saved general csv file and folder with resized images
        csv_filename          - name of general csv file with all coordinates of landmarks
        train_images_dir_name - folder name that will be contain all train images (resized)
        '''

        path_to_merged_csv_file = os.path.join(target_path, csv_filename)
        merged_csv_file = open(path_to_merged_csv_file, 'at+')
        merge_writer = csv.writer(merged_csv_file, delimiter=sep)

        # create directory for new resized images
        dir_target = os.path.join(target_path, train_images_dir_name)
        ensure_dir(dir_target)

        print '\nSTAGE 3: Merging all folders in one, crop all images, merging all csv-files in one csv-file.\n'

        curr_root = ''
        for root, subFolders, files in os.walk(path_to_superdir):
            if root != curr_root:
                curr_root = root
                #print 'Merging: ' + curr_root
            if os.path.exists(os.path.join(curr_root, csv_filename)):
                self.add_one_folder_landmarks(root, dir_target, csv_filename, sep, merged_csv_file, merge_writer)

        merged_csv_file.close()

        print '\n/************************************************************************/'
        print '\nDone: merged all csv-files in one csv-file.'
        print 'Total: {} images.\n'.format(self.cnt_mrg_img)


    def merge_classification(self, path_to_superdir, target_path, train_images_dir_name, landmarks_filename, landmarks_sep, labels_filename, labels_sep):
        '''
        path_to_superdir      - directory where are folders with images and csv-files
        target_path           - directory where will be saved general csv file and folder with resized images
        landmarks_filename    - name of csv file with all coordinates of landmarks
        labels_filename       - name of csv file with labels
        train_images_dir_name - folder name that will be contain all train images (resized)
        '''

        path_to_merged_csv_file = os.path.join(target_path, labels_filename)
        merged_csv_file = open(path_to_merged_csv_file, 'at+')
        merge_writer = csv.writer(merged_csv_file, delimiter=labels_sep)

        # create directory for new resized images
        dir_target = os.path.join(target_path, train_images_dir_name)
        ensure_dir(dir_target)

        print '\nSTAGE 3: Merging all folders in one, crop all images, merging all labels in one csv-file.\n'

        curr_root = ''
        for root, subFolders, files in os.walk(path_to_superdir):
            if root != curr_root:
                curr_root = root
                #print 'Cropping images from dir: ' + curr_root
            if os.path.exists(os.path.join(curr_root, landmarks_filename)) and os.path.exists(os.path.join(curr_root, labels_filename)):
                self.add_one_folder_classification(root, dir_target, landmarks_filename, landmarks_sep, labels_filename, labels_sep, merged_csv_file, merge_writer)

        merged_csv_file.close()

        print '\n/************************************************************************/'
        print '\nDone: merged all folders, cropped and saved all images.'
        print 'Total: {} images.\n'.format(self.cnt_mrg_img)


    def recompute_labels(self, labels, x_low, y_low, transform):

        # print len(labels)
        x = labels[0::2]
        y = labels[1::2]

        # recompute labels
        x -= x_low
        y -= y_low

        array = np.zeros([3, x.shape[0]])
        array[0] = x
        array[1] = y
        array[2] = 1.0
        fp = array.T

        fp = np.transpose(fp)
        fp = transform.dot(fp)
        fp = np.transpose(fp)

        new_labels = np.zeros(labels.shape, dtype='float32')
        new_labels[0::2] = fp.T[0]
        new_labels[1::2] = fp.T[1]

        return new_labels


    def add_one_folder_landmarks(self, dir_src, dir_target, csv_filename, sep, csv_file, writer):

        path_to_initial_csv_file = os.path.join(dir_src, csv_filename)

        # read initial csv
        landmarks = load_landmarks(path_to_initial_csv_file, sep)

        print 'Cropping images from directory {}.'.format(dir_src)

        for idx, row in landmarks.iterrows():
            imgname = row[0].split('/')[-1]
            #print imgname

            # read image
            img = imread(os.path.join(dir_src, imgname))
            #print 'shape: ', img.shape
            imgsize = img.shape[0]

            # read and upscaling labels
            labels = np.array(landmarks.loc[idx, 1:], dtype='float32')

            # crop image
            crop = Crop(img, labels, imgsize, self.imgsize, self.do_shft)
            # if it is needed rescale pts
            crop.rescale_pts()
            # crop
            crop.crop_head()

            # save image
            path_to_img = os.path.join(dir_target, str(self.cnt_mrg_img).zfill(8) + '.jpg')
            crop.save(path_to_img)

            # recompute_labels
            t = crop.create_crop_transform_down(0.5 * crop.new_size)
            new_labels = self.recompute_labels(crop.scaled_img_points, crop.x_offset, crop.y_offset, t)

            # write to scv
            row = [os.path.join(dir_target, str(self.cnt_mrg_img).zfill(8) + '.jpg')]
            row.extend(new_labels)
            writer.writerow(row)
            csv_file.flush()

            self.cnt_mrg_img += 1

        print 'Done: cropped images and csv-files with its labels are created for directory: {}.'.format(dir_src)


    def add_one_folder_classification(self, dir_src, dir_target, landmarks_filename, landmarks_sep, labels_filename, labels_sep, csv_file, writer):

        path_to_landmarks = os.path.join(dir_src, landmarks_filename)
        path_to_labels = os.path.join(dir_src, labels_filename)

        # read initial csv with landmarks
        landmarks = load_landmarks(path_to_landmarks, sep=' ') # sep 'landmarks_sep' was changed by function 'load_cls_landmarks' from load.py

        # read initial csv with labels
        labels = load_cls_labels(path_to_labels, labels_sep, self.task_names)

        # skip NA
        labels, landmarks = labels.dropna(), landmarks.dropna()

        print 'Cropping images from directory {}.'.format(dir_src)

        for idx, row in landmarks.iterrows():
            imgname = row[0].split('/')[-1]
            print imgname

            # read image
            img = imread(os.path.join(dir_src, imgname))
            print 'shape: ', img.shape
            imgsize = img.shape[0]

            # read and upscaling labels
            pts = np.array(landmarks.loc[idx, 1:], dtype='float32')
            #print pts

            # crop image
            crop = Crop(img, pts, imgsize, self.imgsize, self.do_shft)
            # crop
            crop.crop_head()

            # save image
            path_to_img = os.path.join(dir_target, str(self.cnt_mrg_img).zfill(8) + '.jpg')
            crop.save(path_to_img)

            # labels
            imglabels = labels[labels[0] == imgname].values[0].tolist()
            #print imglabels


            # write to scv
            row = [os.path.join(dir_target, str(self.cnt_mrg_img).zfill(8) + '.jpg')]
            row.extend(imglabels[1:]) # without first value of image path
            writer.writerow(row)
            csv_file.flush()

            self.cnt_mrg_img += 1
        print 'Done: cropped images are created and saved for directory: {}.'.format(dir_src)

