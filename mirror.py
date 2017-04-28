#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, csv
from scipy.ndimage import imread
from skimage.io import imsave
import multiprocessing as mp
from  multiprocessing import cpu_count, Pool
from util import ensure_dir, copy_nomatched_file

# CREATE MIRRORS
class Mirror:

    def __init__(self, path_to_superdir, initial_csv_file, new_order):
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
            if not labels.min() < 0.0:
                halfsize = img.shape[0] / 2.
                labels[0::2] = (- (labels[0::2] - halfsize) + halfsize)
            else:
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


    def recursive_create_mirror_images_with_labels(self, queue, new_order, initial_csv_file):
        dir_src = queue.get()
        return self.create_mirror_images_with_labels(dir_src, initial_csv_file, new_order)

    def callback(self):
        print "Working in process #%d" % (os.getpid())

    def run_multiprocessing_mirrors(self):

        # Setup a list of processes that we want to run
        func = self.create_mirror_images_with_labels
        p = Pool(processes=cpu_count())
        for x in range(self.queue.qsize()):
            dir_src = self.queue.get()
            p.apply_async(func=func, args=(dir_src, self.new_order, self.initial_csv_file), callback=self.callback)

        # new section
        p.close()
        p.join()