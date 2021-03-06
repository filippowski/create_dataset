#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import pandas as pd
import os
import fnmatch
import dlib
from skimage import io
from util import is_empty_file
from load import load_landmarks


# CREATE MIRRORS
class Check:

    def __init__(self, path_to_superdir, file_params, mode):
        self.mode = mode

        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)

        if self.mode == 'landmarks':
                self.landmarks_filename     = file_params['in']['landmarks']['csv_filename']
                self.landmarks_names        = file_params['in']['landmarks']['names']
                self.landmarks_types        = file_params['in']['landmarks']['types']
                self.landmarks_sep          = file_params['in']['landmarks']['sep']

        if self.mode == 'classification':
                self.labels_filename        = file_params['in']['labels']['csv_filename']
                self.labels_names           = file_params['in']['labels']['names']
                self.labels_types           = file_params['in']['labels']['types']
                self.labels_sep             = file_params['in']['labels']['sep']

                self.microclasses_filename  = file_params['in']['microclasses']['csv_filename']
                self.microclasses_names     = file_params['in']['microclasses']['names']
                self.microclasses_types     = file_params['in']['microclasses']['types']
                self.microclasses_sep       = file_params['in']['microclasses']['sep']

                self.landmarks_filename     = file_params['in']['landmarks']['csv_filename']
                self.landmarks_names        = file_params['in']['landmarks']['names']
                self.landmarks_types        = file_params['in']['landmarks']['types']
                self.landmarks_sep          = file_params['in']['landmarks']['sep']

                self.path_to_labels         = os.path.join(self.path_to_superdir, self.labels_filename)
                self.path_to_microclasses   = os.path.join(self.path_to_superdir, self.microclasses_filename)
                self.path_to_landmarks      = os.path.join(self.path_to_superdir, self.landmarks_filename)

        if self.mode == '3D':
                self.path_to_alphas         = file_params['in']['alphas']['path_to_alphas']
                self.bunch_fldname          = file_params['in']['alphas']['bunch_fldname']
                self.alphas_fldname         = file_params['in']['alphas']['alphas_fldname']
                self.alphas_ext             = file_params['in']['alphas']['alphas_ext']
                self.alphas_cnt             = file_params['in']['alphas']['alphas_cnt']

                self.path_to_dlib_model     = file_params['in']['dlib_model']['path_to_model']
                self.crop_endswith          = file_params['in']['dlib_model']['crop_endswith']
                self.imgs_ext               = file_params['in']['dlib_model']['imgs_ext']
                self.imgs_cnt               = file_params['in']['dlib_model']['imgs_cnt']

        self.res = True
        self.img_cnt = 0
        self.dir_cnt = 0

    def run(self, mode=None):
        print '\n\nChecking...\n\n'

        if self.mode == 'classification':
            self.checking_classification(self.path_to_superdir)
        if self.mode == 'landmarks':
            self.checking_landmarks(self.path_to_superdir)
        if self.mode == '3D':
            self.checking_3D(self.path_to_superdir)


    def checking_landmarks(self, superdir):

        for root, subFolders, files in os.walk(superdir):
            for f in files:
                if f == self.landmarks_filename:
                    path_to_landmarks = os.path.join(root, f)
                    print 'Checking: {}'.format(path_to_landmarks)
                    try:
                        if os.stat(path_to_landmarks).st_size > 0:
                            print " * full file: {}".format(path_to_landmarks)
                            landmarks = load_landmarks(path_to_landmarks, self.landmarks_sep, self.landmarks_names, self.landmarks_types)
                            for idx, row in landmarks.iterrows():
                                imgname = os.path.split(row[0])[-1]
                                path_to_img = os.path.join(root, imgname)
                                if not os.path.exists(path_to_img):
                                    print 'File not found: {}'.format(path_to_img)
                                    if self.res == True:
                                        self.res = False
                                else:
                                    self.img_cnt += 1
                                if not is_empty_file(path_to_img):
                                    print " * empty file: {}".format(path_to_img)
                        else:
                            print " * empty file: {}".format(path_to_landmarks)
                    except OSError:
                        print " * no file {}".format(path_to_landmarks)

        print 'Check DONE.'
        print 'All right: {}'.format(self.res)
        print 'Count of images in all csv-files: {}'.format(self.img_cnt)


    def checking_classification(self, superdir):

        for root, subFolders, files in os.walk(superdir):
            for f in files:
                if f == self.labels_filename:
                    path_to_labels = os.path.join(root, f)
                    print 'Checking: {}'.format(path_to_labels)
                    try:
                        if os.stat(path_to_labels).st_size > 0:
                            labels = pd.read_csv(path_to_labels, sep=self.labels_sep, header=None)
                            labels = labels.dropna()  # skip skipped images
                            for idx, row in labels.iterrows():
                                imgname = os.path.split(row[0])[-1]
                                path_to_img = os.path.join(root, imgname)
                                if not os.path.exists(path_to_img):
                                    print 'File not found: {}'.format(path_to_img)
                                    if self.res == True:
                                        self.res = False
                                else:
                                    self.img_cnt += 1
                                if is_empty_file(path_to_img):
                                    print " * empty file: {}".format(path_to_img)
                        else:
                            print " * empty file: {}".format(path_to_labels)
                    except OSError:
                        print " * no file {}".format(path_to_labels)

        print 'Check DONE.'
        print 'All right: {}'.format(self.res)
        print 'Count of images in all csv-files: {}'.format(self.img_cnt)


    def checking_3D(self, superdir):

        print 'superdir: ', superdir

        detector = dlib.get_frontal_face_detector()

        for root, subFolders, files in os.walk(superdir):
            for subFolder in subFolders:
                if subFolder[0:5] == self.bunch_fldname:
                    for root_, subFolders_, files_ in os.walk(os.path.join(root, subFolder)):
                        for subFolder_ in subFolders_:

                            #path_to_subFolder_alpha = os.path.join(self.path_to_alphas, subFolder_.split('.obj')[0] + self.alphas_ext)

                            #if not os.path.exists(path_to_subFolder_alpha):
                            #    print 'ALPHA file not found: {}'.format(path_to_subFolder_alpha)
                            #    if self.res == True:
                            #        self.res = False

                            path = os.path.join(root_, subFolder_)
                            imgs_count = len(fnmatch.filter(os.listdir(path), '*'+self.imgs_ext))
                            json_count = len(fnmatch.filter(os.listdir(path), '*'+'.json'))

                            if imgs_count < self.imgs_cnt:
                                print 'In folder {} less than {} images.'.format(subFolder_, self.imgs_cnt)
                                if self.res == True:
                                    self.res = False
                            if json_count < self.imgs_cnt:
                                print 'In folder {} less than {} json.'.format(subFolder_, self.imgs_cnt)
                                if self.res == True:
                                    self.res = False

                            #for img_path in fnmatch.filter(os.listdir(path), '*' + self.imgs_ext):
                            #    path_to_img = os.path.join(path, img_path)
                            #    img = io.imread(path_to_img)
                            #    dets, scores, idx = detector.run(img, 1, -1)
                            #    if len(dets) == 0:
                            #        print "no dets is found for image: {}".format(path_to_img)
                            #        if self.res == True:
                            #            self.res = False
                                #else:
                                    #print "{} dets are found for image: {}".format(len(dets), path_to_img)


                            self.img_cnt += json_count
                            self.dir_cnt += 1
        print 'Check DONE.'
        print 'All right: {}'.format(self.res)
        print 'Count of folders in superdir: {}'.format(self.dir_cnt)
        print 'Count of images in all folders: {}'.format(self.img_cnt)