#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import time
import csv
from scipy.ndimage import imread
from util import Profiler, create_file_with_paths_to_images, ensure_dir, get_alphas_from_alphasfile
from load import load_cls_labels, load_landmarks
from crop import Crop, CropDLIB

# MERGE ALL CSVs IN ONE CSV
class Merge:

    def __init__(self, path_to_superdir, main_path, file_params, crop_params, merge_params, task_params, mode):

        self.mode = mode
        assert self.mode in ['classification', 'landmarks', '3D'], \
            'Mode {} must be one from list {}. Pls check mode param.'.format(self.mode,'[classification, landmarks, 3D]')

        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), \
            'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)

        self.main_path = main_path
        assert os.path.exists(self.main_path), \
            'Path to path where must be saved general csv file and folder with resized images does not exist. Pls check path: {}.'.format(
            self.main_path)

        self.file_params  = file_params
        self.crop_params  = crop_params
        self.merge_params = merge_params
        self.task_params  = task_params

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

        self.labels_filename_out    =  file_params['out']['labels_filename']
        self.images_filename        =  file_params['out']['images_filename']
        self.directory_with_images  =  file_params['out']['directory_with_images']
        self.meanPrefix             =  file_params['out']['meanPrefix']

        self.path_to_dir_with_train_images      =  os.path.join(self.main_path, self.directory_with_images)
        self.path_to_file_with_paths_to_images  =  os.path.join(self.main_path, self.images_filename)
        self.path_to_labels                     =  os.path.join(self.main_path, self.labels_filename_out)

        self.cnt_mrg_img = 0

    def merge(self):

        # merge dataset
        if self.merge_params['merge']:

            with Profiler() as p:

                print '\n\n * merging dataset\n'

                if self.mode == 'classification':
                    print ' * merge classification'
                    self.merge_classification(self.path_to_superdir, self.main_path, self.path_to_dir_with_train_images, self.landmarks_filename, self.landmarks_sep, self.labels_filename, self.labels_sep, self.task_params, self.crop_params)
                if self.mode == 'landmarks':
                    print ' * merge landmarks'
                    self.merge_landmarks(self.path_to_superdir, self.main_path, self.path_to_dir_with_train_images, self.landmarks_filename, self.landmarks_sep, self.crop_params)
                if self.mode == '3D':
                    print ' * merge 3D'
                    self.merge_3D(self.path_to_superdir, self.main_path, self.file_params, self.crop_params)

        # create and save labels
        if self.merge_params['create_labels']:
            with Profiler() as p:
                from labels import Label
                ## labels as numbers
                lbl = Label(self.main_path, self.file_params, self.crop_params, self.task_params, self.mode)
                lbl.create_labels()

                # labels as images
                # create_labels_imgs(main_path, csv_file_name, path_to_labels_imgs)
                # create_file_with_paths_to_images(path_to_labels_imgs, path_to_file_with_paths_to_labels_imgs, img_cnt)

        # create file with images filenames
        if self.merge_params['create_imgfile']:
            with Profiler() as p:
                create_file_with_paths_to_images(self.directory_with_images, self.path_to_file_with_paths_to_images,
                                                 self.path_to_labels)

        # create mean image
        if self.merge_params['create_mean']:
            with Profiler() as p:
                from mean import MeanImage
                mimg = MeanImage(self.main_path, self.path_to_file_with_paths_to_images, self.meanPrefix, self.crop_params)
                mimg.create_mean_image()

                # create_mean_image_by_task(cfg.main_path, cfg.path_to_superdir, cfg.csv_filename, cfg.directory_with_images, cfg.imgSize, cfg.channel)

        # create infogain matrices
        if self.merge_params['create_infogain']:
            with Profiler() as p:
                create_infogain_matrices(self.main_path, self.path_to_superdir, self.csv_filename)

        print '\n/************************************************************************/'
        print 'Done: merged dataset contains {0} images.'.format(self.cnt_mrg_img)


    def merge_landmarks(self, path_to_superdir, main_path, path_to_dir_with_train_images, csv_filename, sep, crop_params):
        '''
        path_to_superdir      - directory where are folders with images and csv-files
        main_path             - directory where will be saved general csv file and folder with resized images
        csv_filename          - name of general csv file with all coordinates of landmarks
        path_to_dir_with_train_images - path to folder that will be contain all train images (resized)
        '''

        path_to_merged_csv_file = os.path.join(main_path, csv_filename)
        merged_csv_file = open(path_to_merged_csv_file, 'at+')
        merge_writer = csv.writer(merged_csv_file, delimiter=sep)

        # create directory for new resized images
        ensure_dir(path_to_dir_with_train_images)

        print '\nSTAGE: Merging all folders in one, crop all images, merging all csv-files in one csv-file.\n'

        curr_root = ''
        for root, subFolders, files in os.walk(path_to_superdir):
            if root != curr_root:
                curr_root = root
                #print 'Merging: ' + curr_root
            if os.path.exists(os.path.join(curr_root, csv_filename)):
                self.add_one_folder_landmarks(root, path_to_dir_with_train_images, csv_filename, sep, merged_csv_file, merge_writer, crop_params)

        merged_csv_file.close()

        print '\n/************************************************************************/'
        print '\nDone: merged all csv-files in one csv-file.'
        print 'Total: {} images.\n'.format(self.cnt_mrg_img)


    def merge_classification(self, path_to_superdir, main_path, path_to_dir_with_train_images, landmarks_filename, landmarks_sep, labels_filename, labels_sep, task_params, crop_params):

        path_to_merged_csv_file = os.path.join(main_path, labels_filename)
        merged_csv_file = open(path_to_merged_csv_file, 'at+')
        merge_writer = csv.writer(merged_csv_file, delimiter=labels_sep)

        # create directory for new resized images
        ensure_dir(path_to_dir_with_train_images)

        print '\nSTAGE: Merging all folders in one, crop all images, merging all labels in one csv-file.\n'

        curr_root = ''
        for root, subFolders, files in os.walk(path_to_superdir):
            if root != curr_root:
                curr_root = root
                print 'Cropping images from dir: ' + curr_root
                print os.path.join(curr_root, landmarks_filename), 'exists: ', os.path.exists(os.path.join(curr_root, landmarks_filename))
                print os.path.join(curr_root, labels_filename), 'exists: ', os.path.exists(os.path.join(curr_root, labels_filename))
            if os.path.exists(os.path.join(curr_root, landmarks_filename)) and os.path.exists(os.path.join(curr_root, labels_filename)):
                self.add_one_folder_classification(root, path_to_dir_with_train_images, landmarks_filename, landmarks_sep, labels_filename, labels_sep, merged_csv_file, merge_writer, task_params, crop_params)

        merged_csv_file.close()

        print '\n/************************************************************************/'
        print '\nDone: merged all folders, cropped and saved all images.'
        print 'Total: {} images.\n'.format(self.cnt_mrg_img)


    def merge_3D(self,  path_to_superdir, main_path, file_params, crop_params):

        print '\nSTAGE: Rewrite alphas.\n'
        # rewrite alphas
        #self.rewrite_alphas()

        print '\nSTAGE: Crope images.\n'
        # crop all images
        crop = CropDLIB(path_to_superdir, file_params, crop_params)

        # run crop
        crop.run_multiprocessing_crop()

        print '\n/************************************************************************/'
        print 'Done: cropped and saved all images.\n\n'


        print '\nSTAGE: Write all alphas in one labels file and all paths to cropped images in images file.\n'

        self.create_imgs_and_lbls_files(crop.nfolders)

        self.cnt_mrg_img = crop.nimgs

        print '\n/************************************************************************/'
        print '\nDone: writed all alphas in one labels file and all paths to cropped images in images file.'
        print 'Total: {} images.\n'.format(crop.nimgs)


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

    def rewrite_alphas(self):
        idx = 0

        for root, subFolders, files in os.walk(self.path_to_superdir):
            for subFolder in subFolders:
                if subFolder[0:5] == self.bunch_fldname:
                    for root_, subFolders_, files_ in os.walk(os.path.join(root, subFolder)):
                        for subFolder_ in subFolders_:

                            path_to_subFolder_alpha = os.path.join(self.path_to_alphas, subFolder_.split('.obj')[0] + self.alphas_ext)

                            if len(subFolder_.split('.obj')[1]) == 0:

                                with open(path_to_subFolder_alpha, 'w') as alphasfile:
                                    alphasfile.write('[' + str(idx) + ']')
                                    alphasfile.close()
                                    print subFolder_.split('.obj')[0], idx
                                idx += 1


    def add_one_folder_landmarks(self, dir_src, dir_target, csv_filename, sep, csv_file, writer, crop_params):

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
            crop = Crop(img, labels, imgsize, crop_params)
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


    def add_one_folder_classification(self, dir_src, dir_target, landmarks_filename, landmarks_sep, labels_filename, labels_sep, csv_file, writer, task_params, crop_params):

        task_names = task_params['tasks_names']

        path_to_landmarks = os.path.join(dir_src, landmarks_filename)
        path_to_labels = os.path.join(dir_src, labels_filename)

        # read initial csv with landmarks
        landmarks = load_landmarks(path_to_landmarks, sep=' ') # sep 'landmarks_sep' was changed by function 'load_cls_landmarks' from load.py

        # read initial csv with labels
        labels = load_cls_labels(path_to_labels, labels_sep, task_names)

        # skip NA
        labels, landmarks = labels.dropna(), landmarks.dropna()

        print 'Cropping images from directory {}.'.format(dir_src)

        for idx, row in landmarks.iterrows():
            imgname = row[0].split('/')[-1]
            #print imgname

            # read image
            img = imread(os.path.join(dir_src, imgname))
            #print 'shape: ', img.shape
            imgsize = img.shape[0]

            # read and upscaling labels
            pts = np.array(landmarks.loc[idx, 1:], dtype='float32')
            #print pts

            # crop image
            crop = Crop(img, pts, imgsize, crop_params)
            # if it is needed rescale pts
            crop.rescale_pts()
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


    def create_imgs_and_lbls_files(self, nfolders):

        idx = 0
        startTime = time.time()

        labels = np.zeros((self.imgs_cnt * nfolders, self.alphas_cnt), dtype='float32')
        file_with_paths_to_images = open(self.path_to_file_with_paths_to_images, "w")

        for root, subFolders, files in os.walk(self.path_to_superdir):
            for subFolder in subFolders:
                if subFolder[0:5] == self.bunch_fldname:
                    for root_, subFolders_, files_ in os.walk(os.path.join(root, subFolder)):
                        for subFolder_ in subFolders_:

                            path_to_subFolder_alpha = os.path.join(self.path_to_alphas, subFolder_.split('.obj')[0] + self.alphas_ext)
                            subFolder_labels = get_alphas_from_alphasfile(path_to_subFolder_alpha, self.alphas_cnt)  # alphas

                            for root1, subFolders1, files1 in os.walk(os.path.join(root_, subFolder_)):

                                file_list = []
                                for f in files1:
                                    filename, file_extension = os.path.splitext(f)
                                    if filename.endswith(self.crop_endswith) and file_extension == self.imgs_ext:
                                        subroot = root1.split(self.main_path)[1]
                                        file_list.append(os.path.join(subroot, f))

                                # for i in range(5):
                                for i in range(len(file_list)):
                                    file_with_paths_to_images.write("%s\n" % file_list[i])
                                    labels[idx] = subFolder_labels
                                    idx += 1

                                # elapsed time report
                                if idx % 1000 == 0:
                                    string_ = str(idx + 1) + ' / ' + str(self.imgs_cnt * nfolders)
                                    sys.stdout.write("\r%s" % string_)
                                    sys.stdout.write(
                                        "\r{}. Elapsed time: {:.3f} sec".format(string_, time.time() - startTime))
                                    sys.stdout.flush()

        # save labels
        np.save(self.path_to_labels, labels)

        print '\nCount of images in all folders: {}'.format(labels.shape[0])
