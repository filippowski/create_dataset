#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, csv
from scipy.ndimage import imread
from skimage.io import imsave
from skimage.transform import rotate
import multiprocessing as mp
from  multiprocessing import cpu_count, Pool
import math
from util import ensure_dir, copy_nomatched_file

# CREATE ROTATIONS
class Rotation:

    def __init__(self, path_to_superdir, initial_csv_file, get_angles):
        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)
        self.get_angles = get_angles
        self.initial_csv_file = initial_csv_file


        # Define folders queue
        self.queue = mp.Queue()
        #m = mp.Manager()
        #self.queue = m.Queue()

        # Put all paths to folders
        for root, subFolders, files in os.walk(self.path_to_superdir):
            if len(subFolders) == 0:
                self.queue.put(root)

        print "queue size: ", self.queue.qsize()
        print self.queue
        print "queue size: ", self.queue.qsize()

    def get_dir_dst_rot(self, dir_src, angles):
        dir_dst = []
        for i in range(len(angles)):
            dir_dst.append(dir_src + '_rotated_' + str(angles[i]))
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


    def rotate_folder(self, dir_src, angles, initial_csv_file):
        print "rotate_folder"
        dir_dst = self.get_dir_dst_rot(dir_src, angles)

        # csv filenames and path-to-files defs
        path_to_initial_csv_file = os.path.join(dir_src, initial_csv_file)
        # print 'Initial csv-file: {}'.format(path_to_initial_csv_file)

        for i in range(len(angles)):

            print 'Rotating by angle {} images from directory {}.'.format(angles[i], dir_src)

            # copy all csv-files that does not matched with initial_csv_file
            copy_nomatched_file(dir_src, dir_dst[i], initial_csv_file)

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

    def recursive_create_rotated_images_with_labels(self, queue, get_angles, initial_csv_file):
        dir_src = queue.get()
        angles = []

        assert callable(get_angles), \
            'Parameter {} must be a function. Pls check params.'.format(get_angles)

        if callable(get_angles):
            print ' * get_angles is function, angles are: {}'.format(get_angles(dir_src))
            angles = get_angles(dir_src)

        return self.rotate_folder(dir_src, angles, initial_csv_file)


    def run_multiprocessing_rotations(self):

        # Setup a list of processes that we want to run
        func = self.recursive_create_rotated_images_with_labels
        args = (self.queue, self.get_angles, self.initial_csv_file)
        processes = [mp.Process(target=func,args=args) for x in range(self.queue.qsize())]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()


# class Rotation_:
#
#     def __init__(self, path_to_superdir, initial_csv_file, get_angles):
#         self.path_to_superdir = path_to_superdir
#         assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)
#         self.get_angles = get_angles
#         self.initial_csv_file = initial_csv_file
#
#         m = mp.Manager()
#         # Define folders queue
#         #self.queue = mp.Queue()
#         self.queue = m.Queue()
#
#         # Put all paths to folders
#         for root, subFolders, files in os.walk(self.path_to_superdir):
#             if len(subFolders) == 0:
#                 self.queue.put(root)
#
#         print "queue size: ", self.queue.qsize()
#         print self.queue
#         print "queue size: ", self.queue.qsize()
#
#
#     def wrap(self, rot):
#         print "wrap"
#         rot.rotate_folder()
#
#
#     def start_process(self):
#         print 'Starting', mp.current_process().name
#
#
#     def callback(self):
#         print "Working in Process #%d" % (os.getpid())
#
#
#     def run_multiprocessing_rotations(self):
#
#         # Setup a list of processes that we want to run
#         p = Pool(processes=cpu_count(), initializer=self.start_process)
#         print "queue size: ", self.queue.qsize()
#         for x in range(self.queue.qsize()):
#             dir_src = self.queue.get()
#             print dir_src
#             angles = []
#             assert callable(self.get_angles), \
#                 'Parameter {} must be a function. Pls check params.'.format(self.get_angles)
#
#             if callable(self.get_angles):
#                 print ' * get_angles is function, angles are: {}'.format(self.get_angles(dir_src))
#                 angles = self.get_angles(dir_src)
#
#             # create object to rotate image from dir_src
#             rot = RotationFolder(dir_src, self.initial_csv_file, angles)
#
#             #p.apply_async(func=func, args=(dir_src, angles, self.initial_csv_file), callback=self.callback)
#             p.apply_async(self.wrap, args=(rot), callback=self.callback)
#             #p.apply_async(rot.rotate_folder, args=(), callback=self.callback)
#
#         # new section
#         p.close()
#         p.join()
