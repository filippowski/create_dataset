#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, re
from util import get_image_size, recompute_row
from config import get_labels_names_and_types

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


def load_cls_labels(filepath, sep, tasks_names, tasks, names=None, types=None):
    raw_tasks_names, work_tasks_names = tasks_names
    raw_labels_names, raw_labels_types = get_labels_names_and_types(raw_tasks_names)
    work_labels_names, _ = get_labels_names_and_types(work_tasks_names)

    # read all raw data from file
    raw_labels = pd.read_csv(filepath, sep=sep, header=None, index_col=False, names=raw_labels_names, dtype=raw_labels_types)

    # get only columns with needed tasks
    labels = raw_labels[work_labels_names]

    # change all values with prefix 'MANUAL' on without it
    labels = labels.applymap(lambda row: re.sub('MANUAL_', '', row))

    # get only rows in labels that have keys in the tasks dictionary
    for key in work_tasks_names:
        labels = labels[labels[key].isin(tasks[key].keys())]

    #if names is not None:
    #    # samples from names and types for only those are in tasks_names
    #    names = [x for x in names if x in tasks_names or names.index(x) == 0]
    #    types = {key: types[key] for key in types.keys() if key in tasks_names or key == names[0]}

    #labels = pd.read_csv(filepath, sep=sep, header=None, index_col=False, names=names, dtype=types)
    #labels = labels[tasks_names]
    #labels = labels.applymap(lambda row: re.sub('MANUAL_', '', row))
    #labels['glasses'] = labels['glasses'].replace('200200', '200100')
    # get only rows in labels that have keys in the tasks dictionary
    #for key in tasks_names:
    #    labels = labels[labels[key].isin(tasks[key].keys())]

    assert labels.isnull().sum().sum() == 0, 'In labels.csv there are NA values: {}! Pls check data.'.format(labels.isnull().sum().sum())
    print ' * labels shape is: {}'.format(labels.shape)
    return labels


def load_cls_landmarks(filepath, sep, names=None, types=None):
    assert 'FILENAME_JPG' in names and 'facepoints' in names, \
        'In landmarks file must be {} and {} columns. Pls check colnames.'.format('FILENAME_JPG', 'facepoints')
    landmarks = pd.read_csv(filepath, sep=sep, header=None, index_col=False, names=names, dtype=types)
    landmarks = landmarks.iloc[:, :3]
    assert landmarks.isnull().sum().sum() == 0, 'In landmarks.csv there are NA values! Pls check data.'
    fnms = pd.DataFrame(landmarks['FILENAME_JPG'])
    fpts = landmarks['facepoints']
    fpts = fpts.apply(str.replace, args=('[', '')).apply(str.replace, args=(']', '')).apply(str.replace, args=(',', ''))
    fpts = fpts.str.split(pat=' ', expand=True)
    # recompute landmarks to interval from -1.0 to 1.0
    root = os.path.split(filepath)[0]
    fnms_fullpath = fnms.applymap(lambda x: os.path.join(root, x))
    fnms_size = fnms_fullpath.applymap(get_image_size)
    fpts_ext = pd.concat([fpts, fnms_size], axis=1)
    fpts_new = fpts_ext.apply(lambda row: recompute_row(row), axis=1).iloc[:, :-1]
    landmarks = pd.concat([fnms, fpts_new], axis=1)
    print ' * landmarks shape is: {}'.format(landmarks.shape)
    return landmarks


def load_cls_microclasses(filepath, sep, names=None, types=None):
    microclasses = pd.read_csv(filepath, sep=sep, header=None, names=names,dtype=types)
    print ' * microclasses shape is: {}'.format(microclasses.shape)
    return microclasses