#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import os
import time
from util import print_cnt
from load import load_cls_labels, load_landmarks

# CREATE AND SAVE LABELS
class Label:

    def __init__(self, main_path, file_params, path_to_labels, mode, task_mask=None, tasks=None, tasks_names=None):
        self.mode = mode
        self.path_to_labels = path_to_labels
        self.main_path = main_path
        assert os.path.exists(self.main_path), \
            'Main path {} does not exist. Pls check path.'.format(self.main_path)

        if self.mode == 'landmarks':
                self.landmarks_filename     = file_params['landmarks']['csv_filename']
                self.landmarks_names        = file_params['landmarks']['names']
                self.landmarks_types        = file_params['landmarks']['types']
                self.landmarks_sep          = file_params['landmarks']['sep']

                self.path_to_raw_landmarks = os.path.join(self.main_path, self.landmarks_filename)
                assert os.path.exists(self.path_to_raw_landmarks), \
                    'Path to labels {} does not exist. Pls check path.'.format(self.path_to_raw_landmarks)

        if self.mode == 'classification':
                self.labels_filename        = file_params['labels']['csv_filename']
                self.labels_names           = file_params['labels']['names']
                self.labels_types           = file_params['labels']['types']
                self.labels_sep             = file_params['labels']['sep']

                self.path_to_raw_labels         = os.path.join(self.main_path, self.labels_filename)
                assert os.path.exists(self.path_to_raw_labels), \
                    'Path to labels {} does not exist. Pls check path.'.format(self.path_to_raw_labels)

        # TO DO
        #if self.mode == '3D':

        self.task_mask = task_mask
        self.tasks = tasks
        self.tasks_names = tasks_names
        self.img_cnt = 0
        self.lbl_cnt = 0
        self.start = time.time()

    def create_labels(self):
        print '\nCreating and saving rotated LABELS file.\n'

        if self.mode == 'classification':
            self.create_labels_classification()
        if self.mode == 'landmarks':
            self.create_labels_landmarks()
        if self.mode == '3D':
            print "to be .. TO DO"
            #    self.run_augmentation_3D()

        print '\n/************************************************************************/'
        print 'Done: created and saved LABELS file.\n\n'

        print 'Merged dataset contains {0} images with {1} labels.'.format(self.img_cnt, self.lbl_cnt)


    def get_labels_length(self, tasks_names):
        cnt = 0
        for el in tasks_names:
            cnt += 1
        print cnt, len(tasks_names)
        return cnt

    def get_mask(self):
        return np.array([1, 1, 1, 0, 0, 0, 0,
                         1, 1, 0, 0, 0, 0, 0,
                         1, 1, 1, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 0, 0, 0,
                         1, 1, 1, 1, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 0], dtype='int32')

    def create_labels_classification(self):
        labels_length = self.get_labels_length(self.tasks_names[1])
        # add if it is needed size of mask to labels length
        if self.task_mask:
            mask = get_mask()
            labels_length = labels_length + mask.size

        raw_labels = load_cls_labels(self.path_to_raw_labels, self.labels_sep, self.tasks_names[0])
        print ' * raw labels: ', raw_labels.iloc[0]
        # get only needed labels
        columns = [x for x in self.labels_names if x in self.tasks_names[1] or self.labels_names.index(x) == 0]
        cut_labels = raw_labels[columns]
        print ' * cut labels: ', cut_labels.iloc[0]

        labels = np.zeros((cut_labels.shape[0], labels_length), dtype='int32')

        cnt_lbl = 0
        for idx, row in cut_labels.iterrows():
            print_cnt(cnt_lbl, cut_labels.shape[0], 1000, self.start)
            lbls = np.array([], dtype='int32')
            for i in self.tasks_names[1]:
                # print i, row[i]
                lbls = np.append(lbls, self.tasks[i][row[i]])
            if self.task_mask:
                lbls = np.append(lbls, mask_array)
            # print lbls
            labels[cnt_lbl] = lbls
            cnt_lbl += 1

        np.save(self.path_to_labels, labels)
        self.img_cnt, self.lbl_cnt = labels.shape

    def create_labels_landmarks(self):
        raw_landmarks = load_landmarks(self.path_to_raw_landmarks, self.landmarks_sep, self.landmarks_names, self.landmarks_types)
        landmarks = np.array(raw_landmarks.iloc[:,1:], dtype='float32')
        np.save(self.path_to_labels, landmarks)
        self.img_cnt, self.lbl_cnt = landmarks.shape


# TO DO refactoring
'''
    def create_labels_imgs(main_path, csv_file_name, path_to_labels_imgs):
        general_landmarks = pd.read_csv(os.path.join(main_path, csv_file_name), sep=' ', header=None)
        labels = np.array(general_landmarks.loc[:, 1:], dtype='float32')
        img_cnt, lbl_cnt = labels.shape

        # save labels as images
        for k in range(labels.shape[0]):
            pts = labels[k]
            x1_coor, y1_coor = get_coords(pts, 112)
            label_img = np.zeros((224, 224))
            imgnm = general_landmarks.loc[k, 0].split('/')[-1]
            for i in range(len(x1_coor)):
                color = int(round(2.5 * i + 42))
                cv2.circle(label_img, tuple([int(x1_coor[i]), int(y1_coor[i])]), 0, (color), 1)
            imsave(os.path.join(path_to_labels_imgs, imgnm), (label_img) / 256.)

        return img_cnt, lbl_cnt
'''


