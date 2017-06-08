#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, csv
import glob
from scipy.ndimage import imread
from skimage.io import imsave
from skimage.transform import rotate
import multiprocessing as mp
from  multiprocessing import cpu_count, Pool
import math
from util import ensure_dir, copy_auxiliary_files

# CREATE ROTATIONS
class Rotation:

    def __init__(self, path_to_superdir, get_angles, initial_csv_file=None):
        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)
        self.get_angles = get_angles
        self.initial_csv_file = initial_csv_file

        # Define folders queue
        self.queue = mp.Queue()

        # Put all paths to folders
        for root, subFolders, files in os.walk(self.path_to_superdir):
            if len(subFolders) == 0:
                self.queue.put(root)


    def get_dir_dst_rot(self, dir_src, angles):
        dir_dst = []
        for i in range(len(angles)):
            dir_dst.append('{}_rotated_{}_{}'.format(dir_src, str(angles[i]), str(i)))
            ensure_dir(dir_dst[i])
        return dir_dst

    def create_rotate_transform(self, angle, center):
        rotate = np.identity(3)
        rotate[0, 0] = math.cos(angle)
        rotate[0, 1] = -math.sin(angle)
        rotate[1, 0] = math.sin(angle)
        rotate[1, 1] = math.cos(angle)

        translate_to_origin = np.identity(3)
        translate_to_origin[0, 2] = -center[0]
        translate_to_origin[1, 2] = -center[1]

        translate_back = np.identity(3)
        translate_back[0, 2] = center[0]
        translate_back[1, 2] = center[1]

        transform = np.identity(3)
        transform = transform.dot(translate_back)
        transform = transform.dot(rotate)
        transform = transform.dot(translate_to_origin)
        return transform


    def create_rotate_shift_transform(self, angle, center, shift):
        rotate = np.identity(3)
        rotate[0, 0] = math.cos(angle)
        rotate[0, 1] = -math.sin(angle)
        rotate[1, 0] = math.sin(angle)
        rotate[1, 1] = math.cos(angle)
        rotate[0, 2] = shift[0]
        rotate[1, 2] = shift[1]

        translate_to_origin = np.identity(3)
        translate_to_origin[0, 2] = -center[0]
        translate_to_origin[1, 2] = -center[1]

        translate_back = np.identity(3)
        translate_back[0, 2] = center[0]
        translate_back[1, 2] = center[1]

        transform = np.identity(3)
        transform = transform.dot(translate_back)
        transform = transform.dot(rotate)
        transform = transform.dot(translate_to_origin)
        return transform


    def create_rotated_images_with_labels(self, dir_src, angles, initial_csv_file):
        dir_dst = self.get_dir_dst_rot(dir_src, angles)

        # csv filenames and path-to-files defs
        path_to_initial_csv_file = os.path.join(dir_src, initial_csv_file)
        # print 'Initial csv-file: {}'.format(path_to_initial_csv_file)

        for i in range(len(angles)):

            print 'Rotating by angle {} images from directory {}.'.format(angles[i], dir_src)

            # copy all csv-files that does not matched with initial_csv_file
            copy_auxiliary_files(dir_src, dir_dst[i], needless_files_names=(initial_csv_file), extensions=('.csv'))

            path_to_rotated_csv_file = os.path.join(dir_dst[i], initial_csv_file)
            # print path_to_rotated_csv_file

            # new csv file
            rotated_csv_file = open(path_to_rotated_csv_file, 'wt+')
            rotated_writer = csv.writer(rotated_csv_file, delimiter=' ')

            # read initial csv
            general_landmarks = pd.read_csv(path_to_initial_csv_file, sep=' ', header=None)
            general_landmarks = general_landmarks.dropna()  # skip skipped images
            for idx, row in general_landmarks.iterrows():
                imgname = row[0].split('/')[-1]
                # print imgname
                # save images
                img = imread(os.path.join(dir_src, imgname))
                img_rotated = rotate(img, angles[i], mode='symmetric')
                path_to_img = os.path.join(dir_dst[i], imgname)
                imsave(path_to_img, img_rotated)

                # save labels
                labels = np.array(general_landmarks.loc[idx, 1:], dtype='float32')
                # print len(labels)
                x = labels[0::2]
                y = labels[1::2]

                array = np.zeros([3, x.shape[0]])
                array[0] = x
                array[1] = y
                array[2] = 1.0
                fp = array.T

                t = self.create_rotate_transform(-angles[i] / 57.2958, (0.0, 0.0))
                fp = np.transpose(fp)
                fp = t.dot(fp)
                fp = np.transpose(fp)

                new_labels = np.zeros(labels.shape, dtype='float32')
                new_labels[0::2] = fp.T[0]
                new_labels[1::2] = fp.T[1]

                row = [os.path.join(dir_dst[i], imgname)]
                row.extend(new_labels)
                rotated_writer.writerow(row)
                rotated_csv_file.flush()

            rotated_csv_file.close()
        print 'Done: rotated images and csv-files with its labels are created for directory: {}.'.format(dir_src)


    def create_rotated_images_wo_labels(self, dir_src, angles):
        dir_dst = self.get_dir_dst_rot(dir_src, angles)

        for i in range(len(angles)):

            # copy all files that are needed
            copy_auxiliary_files(dir_src, dir_dst[i], extensions=('.json'))

            print 'Rotating by angle {} images from directory {}.'.format(angles[i], dir_src)

            for f in glob.glob(os.path.join(dir_src, '*' + '.jpg')):

                # save images
                img = imread(f)
                img_rotated = rotate(img, angles[i], mode='symmetric')
                path_to_img = os.path.join(dir_dst[i], os.path.split(f)[1])
                imsave(path_to_img, img_rotated)

        print 'Done: rotated images and csv-files with its labels are created for directory: {}.'.format(dir_src)


    def recursive_create_rotated_images_with_labels(self, queue, get_angles, initial_csv_file):
        dir_src = queue.get()
        angles = []

        assert callable(get_angles), \
            'Parameter {} must be a function. Pls check params.'.format(get_angles)

        if callable(get_angles):
            angles = get_angles(dir_src)
            #print ' * get_angles is function, angles are: {}'.format(angles)

        if initial_csv_file is None:
            #print 'Rotate images w/o labels.'
            self.create_rotated_images_wo_labels(dir_src, angles)
        else:
            #print 'Rotate images with labels.'
            self.create_rotated_images_with_labels(dir_src, angles, initial_csv_file)


    def run_multiprocessing_rotations(self):

        # Setup a list of processes that we want to run
        func = self.recursive_create_rotated_images_with_labels
        args = (self.queue, self.get_angles, self.initial_csv_file)
        processes = [mp.Process(target=func,args=args) for x in range(self.queue.qsize())]

        nprocesses = len(processes)
        nworkers = int(0.75*mp.cpu_count())

        for i in range(int(nprocesses/nworkers)+1):
            proc = processes[:nworkers]
            processes = processes[nworkers:]

            # Run processes
            for p in proc:
                p.start()

            # Exit the completed processes
            for p in proc:
                p.join()