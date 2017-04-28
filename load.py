#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from util import labels_str_to_flatten_list

def loading(csv_file):
    list_processed_img = []
    general_landmarks = pd.read_csv(csv_file, sep=' ', header=None)
    for index, row in general_landmarks.iterrows():
        list_processed_img.append(row[0].split('/')[-1])
    labels = np.array(general_landmarks.loc[:, 1:], dtype='float32')

    return labels, list_processed_img

def load_json(jsonfile):
    import json
    with open(jsonfile, 'r') as f:
        predictions = json.load(f)
    #
    # # predicted points
    # face_pts = predictions['face_points']
    return predictions

def load_landmarks(filepath, sep, names=None, types=None):
    landmarks = pd.read_csv(filepath, sep=sep, header=None, names=names,dtype=types)
    landmarks = landmarks.dropna()  # skip skipped images
    return landmarks


def load_cls_labels(filepath, sep, names=None, types=None):
    labels = pd.read_csv(filepath, sep=sep, header=None, names=names,dtype=types)
    labels = labels.dropna()
    # print ' * labels shape is: {}'.format(labels.shape)
    return labels

def load_cls_landmarks(filepath, sep, names=None, types=None):
    assert 'FILENAME_JPG' in names and 'facepoints' in names, \
        'In landmarks file must be {} and {} columns. Pls check colnames.'.format('FILENAME_JPG', 'facepoints')
    landmarks = pd.read_csv(filepath, sep=sep, header=None, index_col=False, names=names, dtype=types)
    fnms = pd.DataFrame(landmarks['FILENAME_JPG'])
    fpts = landmarks['facepoints']
    fpts = fpts.apply(str.replace, args=('[', '')).apply(str.replace, args=(']', '')).apply(str.replace, args=(',', ''))
    fpts = fpts.str.split(pat=' ', expand=True)
    landmarks = pd.concat([fnms, fpts], axis=1)
    # print ' * landmarks shape is: {}'.format(landmarks.shape)
    return landmarks

def load_cls_microclasses(filepath, sep, names, types):
    microclasses = pd.read_csv(filepath, sep=sep, header=0, names=names,dtype=types)
    # print ' * microclasses shape is: {}'.format(microclasses.shape)
    return microclasses