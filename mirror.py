#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, csv
import glob
from scipy.ndimage import imread
from skimage.io import imsave
import multiprocessing as mp
from  multiprocessing import cpu_count, Pool
from util import ensure_dir, copy_nomatched_file

# CREATE MIRRORS
class Mirror:

    def __init__(self, path_to_superdir, new_order=None, initial_csv_file=None):
        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)
        self.new_order = new_order
        self.initial_csv_file = initial_csv_file

        # Define folders queue
        self.queue = mp.Queue()

        # Put all paths to folders
        for root, subFolders, files in os.walk(self.path_to_superdir):
            if len(subFolders) == 0:
                self.queue.put(root)

    def get_dir_dst_mir(self, dir_src):
        dir_dst = dir_src + '_mirror'
        ensure_dir(dir_dst)
        return dir_dst

    def create_mirror_images_with_labels(self, dir_src, new_order, csv_filename):
        # create dst directory
        dir_dst = self.get_dir_dst_mir(dir_src)

        print 'Mirror images from directory {}.'.format(dir_src)

        # copy all csv-files that does not matched with csv_filename
        copy_nomatched_file(dir_src, dir_dst, csv_filename)

        # csv filenames and path-to-files defs
        path_to_initial_csv_file = os.path.join(dir_src, csv_filename)
        path_to_mirrored_csv_file = os.path.join(dir_dst, csv_filename)

        # new csv file
        mirrored_csv_file = open(path_to_mirrored_csv_file, 'wt+')
        mirrored_writer = csv.writer(mirrored_csv_file, delimiter=' ')

        # read initial csv
        general_landmarks = pd.read_csv(path_to_initial_csv_file, sep=' ', header=None)
        general_landmarks = general_landmarks.dropna()  # skip skipped images
        for idx, row in general_landmarks.iterrows():
            imgname = row[0].split('/')[-1]
            # print imgname

            # save images
            img = imread(os.path.join(dir_src, imgname))
            img_mirror = np.fliplr(img)
            path_to_img = os.path.join(dir_dst, imgname)
            imsave(path_to_img, img_mirror)

            # save labels
            labels = np.array(general_landmarks.loc[idx, 1:], dtype='float32')
            # print len(labels)

            labels[0::2] = -labels[0::2]  # inverse labels across x-axis

            new_labels = np.zeros(labels.shape)
            for i in range(len(new_order)):
                new_labels[2 * new_order[i]] = labels[2 * i]
                new_labels[2 * new_order[i] + 1] = labels[2 * i + 1]
            row = [os.path.join(dir_dst, imgname)]
            row.extend(new_labels)
            mirrored_writer.writerow(row)
            mirrored_csv_file.flush()

        mirrored_csv_file.close()
        print 'Done: mirrored images and csv-files with its labels are created for directory: {}.'.format(dir_src)


    def create_mirror_images_wo_labels(self, dir_src):
        # create dst directory
        dir_dst = self.get_dir_dst_mir(dir_src)

        print 'Mirror images from directory {}.'.format(dir_src)

        for f in glob.glob(os.path.join(dir_src, '*' + '.jpg')):

            # save images
            img = imread(f)
            img_mirror = np.fliplr(img)
            path_to_img = os.path.join(dir_dst, os.path.split(f)[1])
            imsave(path_to_img, img_mirror)

        print 'Done: mirrored images are created for directory: {}.'.format(dir_src)


    def recursive_create_mirror_images_with_labels(self, queue, new_order, initial_csv_file):
        dir_src = queue.get()

        if initial_csv_file is None:
            self.create_mirror_images_wo_labels(dir_src)
        else:
            self.create_mirror_images_with_labels(dir_src, new_order, initial_csv_file)


    def run_multiprocessing_mirrors(self):

        # Setup a list of processes that we want to run
        func = self.recursive_create_mirror_images_with_labels
        args = (self.queue, self.new_order, self.initial_csv_file)
        processes = [mp.Process(target=func,args=args) for x in range(self.queue.qsize())]

        nprocesses = len(processes)
        #print nprocesses
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