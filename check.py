#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import pandas as pd
import os
from util import is_empty_file

# CREATE MIRRORS
class Check:

    def __init__(self, path_to_superdir, mode):
        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)
        self.mode = mode
        self.res = True
        self.cnt = 0


    def run(self, mode=None):
        if self.mode == 'classification':
            self.checking_classification(self.path_to_superdir)
        if self.mode == 'landmarks':
            self.checking_landmarks(self.path_to_superdir)
        #if self.mode == '3D':
        #    self.checking_3D(self.path_to_superdir)

    def checking_classification(self, superdir):

        for root, subFolders, files in os.walk(superdir):
            for f in files:
                filename, file_extension = os.path.splitext(f)
                if filename == 'labels' and file_extension == '.csv':
                    path_to_csv_file = os.path.join(root, 'labels.csv')
                    print 'Checking: {}'.format(path_to_csv_file)
                    try:
                        if os.stat(path_to_csv_file).st_size > 0:
                            labels = pd.read_csv(path_to_csv_file, sep=';', header=None)
                            labels = labels.dropna()  # skip skipped images
                            for idx, row in labels.iterrows():
                                imgname = row[0].split('/')[-1]
                                path_to_img = os.path.join(root, imgname)
                                path_to_json = os.path.join(root, imgname.split('.')[0] + ".json")
                                if not os.path.exists(path_to_img):
                                    print 'File not found: {}'.format(path_to_img)
                                    if self.res == True:
                                        self.res = False
                                else:
                                    self.cnt += 1
                                if is_empty_file(path_to_img):
                                    print " * empty file: {}".format(path_to_img)
                        else:
                            print " * empty file: {}".format(path_to_csv_file)
                    except OSError:
                        print " * no file {}".format(path_to_csv_file)

        print 'Check DONE.'
        print 'All right: {}'.format(self.res)
        print 'Count of images in all csv-files: {}'.format(self.cnt)


    def checking_landmarks(self, superdir):

        for root, subFolders, files in os.walk(superdir):
            for f in files:
                filename, file_extension = os.path.splitext(f)
                if filename == 'landmarks' and file_extension == '.csv':
                    path_to_csv_file = os.path.join(root, 'landmarks.csv')
                    print 'Checking: {}'.format(path_to_csv_file)
                    try:
                        if os.stat(path_to_csv_file).st_size > 0:
                            print " * full file: {}".format(path_to_csv_file)
                            landmarks = pd.read_csv(path_to_csv_file, sep=' ', header=None)
                            landmarks = landmarks.dropna()  # skip skipped images
                            for idx, row in landmarks.iterrows():
                                imgname = row[0].split('/')[-1]
                                path_to_img = os.path.join(root, imgname)
                                if not os.path.exists(path_to_img):
                                    print 'File not found: {}'.format(path_to_img)
                                    if self.res == True:
                                        self.res = False
                                else:
                                    self.cnt += 1
                                if not is_empty_file(path_to_img):
                                    print " * empty file: {}".format(path_to_img)
                        else:
                            print " * empty file: {}".format(path_to_csv_file)
                    except OSError:
                        print " * no file {}".format(path_to_csv_file)

        print 'Check DONE.'
        print 'All right: {}'.format(self.res)
        print 'Count of images in all csv-files: {}'.format(self.cnt)