#!/usr/bin/env python2
## -*- coding: utf-8 -*-


import os, csv
import multiprocessing as mp
import shutil
from rotation import Rotation
from mirror import Mirror
from util import ensure_dir, remove, clean_csv
from load import load_cls_labels, load_cls_landmarks, load_cls_microclasses

# AUGMENTATION DATASET
class Augmentation:

    def __init__(self, path_to_superdir, file_params, augmentation_params, mode):

        self.mode = mode
        self.path_to_superdir = path_to_superdir

        assert os.path.exists(self.path_to_superdir), \
            'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)

        assert self.mode in ['classification', 'landmarks', '3D'], \
            'Mode {} must be one from list {}. Pls check mode param.'.format(self.mode,'[classification, landmarks, 3D]')

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

                assert os.path.exists(self.path_to_labels), \
                    'Path to labels {} does not exist. Pls check path.'.format(self.path_to_labels)
                assert os.path.exists(self.path_to_microclasses), \
                    'Path to microclasses {} does not exist. Pls check path.'.format(self.path_to_microclasses)
                assert os.path.exists(self.path_to_landmarks), \
                    'Path to landmarks {} does not exist. Pls check path.'.format(self.path_to_landmarks)

        # TO DO
        #if self.mode == '3D':

        self.distortion = augmentation_params['distortion']['do']
        self.schemes = augmentation_params['distortion']['schemes']
        self.rotation = augmentation_params['rotation']['do']
        self.angles = augmentation_params['rotation']['angles']
        self.mirror = augmentation_params['mirror']['do']
        self.new_order = augmentation_params['mirror']['new_order']

        # get right function to get angles for rotation
        self.angles = self.angles(self.mode)


    def run_augmentation(self):
        if self.mode == 'classification':
            self.run_augmentation_classification()
        if self.mode == 'landmarks':
            self.run_augmentation_landmarks()
        if self.mode == '3D':
            print "to be .. TO DO"
        #    self.run_augmentation_3D()


    def run_augmentation_landmarks(self):

        if self.distortion:
            print '\n * distortion\n'

            # run distortion
            # TO DO..

        if self.rotation:
            print '\n * rotation\n'
            rot = Rotation(self.path_to_superdir, self.landmarks_filename, self.angles)

            # run rotation

            print '\nSTAGE 1: Creating and saving rotated images and csv-files with its labels:\n'

            rot.run_multiprocessing_rotations()

            print '\n/************************************************************************/'
            print 'Done: created and saved rotated images and csv-files with its labels.\n\n'

        if self.mirror:
            print '\n * mirror\n'
            mir = Mirror(self.path_to_superdir, self.landmarks_filename, self.new_order)

            # run mirror

            print '\nSTAGE 2: Creating and saving mirrored images and csv-files with its labels:\n'

            mir.run_multiprocessing_mirrors()

            print '\n/************************************************************************/'
            print 'Done: created and saved mirrored images and csv-files with its labels.\n\n'

    def run_augmentation_classification(self):

        # divide the dataset into microclasses folders
        self.divide_dataset_to_microclass_folders()

        if self.distortion:
            print '\n * distortion\n'

            # run distortion
            # TO DO..

        if self.rotation:
            print '\n * rotation\n'
            rot = Rotation(self.path_to_superdir, self.landmarks_filename, self.angles)

            # run rotation

            print '\nSTAGE 1: Creating and saving rotated images and csv-files with its labels:\n'

            rot.run_multiprocessing_rotations()

            print '\n/************************************************************************/'
            print 'Done: created and saved rotated images and csv-files with its labels.\n\n'

        if self.mirror:
            print '\n * mirror\n'
            mir = Mirror(self.path_to_superdir, self.landmarks_filename, self.new_order)

            # run mirror

            print '\nSTAGE 2: Creating and saving mirrored images and csv-files with its labels:\n'

            mir.run_multiprocessing_mirrors()

            print '\n/************************************************************************/'
            print 'Done: created and saved mirrored images and csv-files with its labels.\n\n'


    def divide_dataset_to_microclass_folders(self):
        # read initial csv with labels
        dataset = load_cls_labels(self.path_to_labels, self.labels_sep, self.labels_names, self.labels_types)

        # read initial csv with microclasses
        microclasses = load_cls_microclasses(self.path_to_microclasses, self.microclasses_sep, self.microclasses_names, self.microclasses_types)

        # read initial csv with landmarks
        landmarks = load_cls_landmarks(self.path_to_landmarks, self.landmarks_sep, self.landmarks_names, self.landmarks_types)

        # skip NA
        dataset, microclasses, landmarks = dataset.dropna(), microclasses.dropna(), landmarks.dropna()

        # run across rows in microclasses table and divide superdir into classes
        for micro_idx, micro_row in microclasses.iterrows():
            # new folder
            new_folder_name = self.get_folder_name(micro_idx, micro_row, self.microclasses_names)
            path_to_new_folder = os.path.join(self.path_to_superdir, new_folder_name)
            ensure_dir(path_to_new_folder)

            # new csv files
            path_to_new_labels_file = os.path.join(path_to_new_folder, self.labels_filename)
            new_labels_file   = open(path_to_new_labels_file, 'wt+')
            new_labels_writer = csv.writer(new_labels_file, delimiter=self.labels_sep)

            path_to_new_landmarks_file = os.path.join(path_to_new_folder, self.landmarks_filename)
            new_landmarks_file = open(path_to_new_landmarks_file, 'wt+')
            new_landmarks_writer = csv.writer(new_landmarks_file, delimiter=' ')

            for f in micro_row['filenames_list'].split(' '):
                # copy image in new folder
                path_src = os.path.join(self.path_to_superdir, f)
                path_dst = os.path.join(path_to_new_folder, f)
                shutil.copy2(path_src, path_dst)
                # remove original file
                remove(path_src)

                # add row with labels in new labels csv file
                labels_row = dataset[dataset[self.labels_names[0]] == f]
                new_labels_writer.writerow(labels_row.values.tolist()[0])

                # add row with labels in new labels csv file
                landmarks_row = landmarks[landmarks[self.landmarks_names[0]] == f]
                new_landmarks_writer.writerow(landmarks_row.values.tolist()[0])

                # flush new csv files
                new_labels_file.flush(), new_landmarks_file.flush()

            # close new csv files
            new_labels_file.close(), new_landmarks_file.close()

        # move original scv_file in main path and cleaning it
        new_path_to_labels = os.path.abspath(os.path.join(os.path.dirname(self.path_to_labels), os.pardir, self.labels_filename))
        shutil.move(self.path_to_labels, new_path_to_labels)
        clean_csv(new_path_to_labels)
        # renew path to labels
        self.path_to_labels = new_path_to_labels

        # remove landmarks file
        remove(self.path_to_landmarks)

        print 'Done: the dataset was divided into {} microclasses folders.'.format(microclasses.shape[0])

    def get_folder_name(self, idx, row, keys):
        folder_name = str(idx)
        for i in range(len(keys)-2):
            folder_name += '_{}'.format(row[keys[i]])
        return folder_name

    def get_queue(self, path):
        # Define folders queue
        queue = mp.Queue()

        # Put all paths to folders
        for root, subFolders, files in os.walk(path):
            if len(subFolders) == 0:
                queue.put(root)
        return queue