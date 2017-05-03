#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import os
from newOrder import get_new_order_119

################################################
#  TODO set flags
################################################
# augmentation dataset
augmentation = True
# merge all csv in one
merge = False
# create and save labels
create_labels = False
# create file with images filenames
create_imgfile = False

# create mean image
create_mean = False
# create infogain matrices
create_infogain = False

################################################
#  TODO set path_names
################################################
# Full path to directory 'superdir' that contain some count of folders with images and 'landmarks.csv' files

main_path = '/home/filippovski/deep-learning/script_test/cls'
mode = 'classification'

assert mode in ['classification', 'landmarks', '3D'], \
    'Mode {} must be one from list {}. Pls check mode param.'.format(mode, '[classification, landmarks, 3D]')


images_filename = 'images.txt'
labels_filename = 'labels.npy'
directory_with_images = 'train'
meanPrefix = 'mean'

csv_filename = 'labels.csv' if mode == 'classification' else 'landmarks.csv'
microclasses_filename = 'microclasses.csv'

path_to_superdir = os.path.join(main_path, 'superdir')
path_to_labels = os.path.join(main_path, labels_filename)
path_to_file_with_paths_to_images = os.path.join(main_path, images_filename)
path_to_dir_with_images = os.path.join(main_path, directory_with_images)
path_to_lmdb_with_images = os.path.join(main_path, 'lmdb_images')
path_to_lmdb_with_labels = os.path.join(main_path, 'lmdb_labels')

imgSize  = 224  # width and height size of image (image must be same width and height size)
channel  = 3    # channels number of images
testSize = 20   # percentage of test examples from whole dataset


################################################
#  TODO set params
################################################
# углы поворотов для rotation
angles = [3, 6]
#angles = [3, 6, 9, 12, 15, 18, 21]
#angles = range(1, 60, 3)
#angles = range(1, 40, 2)


################################################
#  TODO set classes
################################################

# do shift during images cropping
do_shft = True
# write to labels their mask in multitask classification
task_mask = False

# разделитель лейблов в landmarks.csv
landmarks_sep    = ' '
# разделитель лейблов в labels.csv
labels_sep       = ';'
# разделитель лейблов в microclasses.csv
microclasses_sep = ' '


# названия лейблов в labels.csv
labels_names = [
                'FILENAME_JPG',
                'skin',
                'gender',
                'hair_cover',
                'hair_color',
                'hair_len',
                'hair_type',
                'hair_fringe',
                'beard',
                'glasses',
                'face',
                'mouth',
                'nose',
                'face_exp',
                'brows'
                ]

# типы лейблов в labels.csv
labels_types = {
                 'FILENAME_JPG': str,
                 'skin':         str,
                 'gender':       str,
                 'hair_cover':   str,
                 'hair_color':   str,
                 'hair_len':     str,
                 'hair_type':    str,
                 'hair_fringe':  str,
                 'beard':        str,
                 'glasses':      str,
                 'face':         str,
                 'mouth':        str,
                 'nose':         str,
                 'face_exp':     str,
                 'brows':        str
                }


# названия лейблов в labels.csv
landmarks_names = [
                'FILENAME_JPG',
                'bbox',
                'facepoints'
                ]

# типы лейблов в labels.csv
landmarks_types = {
                 'FILENAME_JPG': str,
                 'bbox':         str,
                 'facepoints':   str
                }

# названия лейблов в microclasses.csv
microclasses1_names = [
                      'skin',
                      'gender',
                      'hair_cover',
                      'hair_color',
                      'hair_len',
                      'hair_type',
                      'hair_fringe',
                      'beard',
                      'glasses',
                      'count',
                      'filenames_list'
                     ]
# названия лейблов в microclasses.csv
microclasses1_types = {
                      'skin':           str,
                      'gender':         str,
                      'hair_cover':     str,
                      'hair_color':     str,
                      'hair_len':       str,
                      'hair_type':      str,
                      'hair_fringe':    str,
                      'beard':          str,
                      'glasses':        str,
                      'count':          int,
                      'filenames_list': str
                     }


# названия лейблов в microclasses.csv
microclasses2_names = [
                      'face',
                      'mouth',
                      'nose',
                      'face_exp',
                      'brows',
                      'count',
                      'filenames_list'
                     ]
# названия лейблов в microclasses.csv
microclasses2_types = {
                        'face':             str,
                        'mouth':            str,
                        'nose':             str,
                        'face_exp':         str,
                        'brows':            str,
                        'count':            int,
                        'filenames_list':   str
                     }

################################################
#  TODO set params
################################################

# choose params for reading microclasses file (type 1 or 2)
microclasses_names = microclasses2_names
microclasses_types = microclasses2_types

################################################

# TODO fill true tasks digits labels
def get_tasks():
    tasks = {
        'skin': {
            'white':        np.array([0], dtype='int32'),
            'dark':         np.array([1], dtype='int32'),
            'asian':        np.array([2], dtype='int32')
        },

        'gender': {
            'male':         np.array([0], dtype='int32'),
            'female':       np.array([1], dtype='int32')
        },

        'hair_cover': {
            'no':           np.array([0], dtype='int32'),
            'hijab':        np.array([1], dtype='int32'),
            'hat':          np.array([2], dtype='int32')
        },

        'hair_color': {
            'black':        np.array([0], dtype='int32'),
            'brown':        np.array([1], dtype='int32'),
            'light-brown':  np.array([2], dtype='int32'),
            'blond':        np.array([3], dtype='int32'),
            'carroty':      np.array([4], dtype='int32'),
            'grey':         np.array([5], dtype='int32'),
            'undefined':    np.array([6], dtype='int32')
        },

        'hair_len': {
            '1':            np.array([0], dtype='int32'),
            '2':            np.array([1], dtype='int32'),
            '3':            np.array([2], dtype='int32'),
            '4':            np.array([3], dtype='int32'),
            '5':            np.array([4], dtype='int32'),
            '6':            np.array([5], dtype='int32'),
            'undefined':    np.array([6], dtype='int32')
        },

        'hair_type': {
            'curly':        np.array([0], dtype='int32'),
            'straight':     np.array([1], dtype='int32'),
            'wavy':         np.array([2], dtype='int32'),
            'undefined':    np.array([3], dtype='int32')
        },

        'hair_fringe': {
            'close':        np.array([0], dtype='int32'),
            'open':         np.array([1], dtype='int32'),
            'partial':      np.array([2], dtype='int32'),
            'undefined':    np.array([3], dtype='int32')
        },

        'beard': {
            'without':      np.array([0], dtype='int32'),
            'bristle':      np.array([1], dtype='int32'),
            'mustache':     np.array([2], dtype='int32'),
            'circle_beard': np.array([3], dtype='int32'),
            'full_beard':   np.array([4], dtype='int32'),
            'just_beard':   np.array([5], dtype='int32'),
            'balbo':        np.array([6], dtype='int32')
        },

        'glasses': {
            '100000':       np.array([0], dtype='int32'),
            '100200':       np.array([1], dtype='int32'),
            '200100':       np.array([2], dtype='int32'),
            '300100':       np.array([3], dtype='int32'),
            '400100':       np.array([4], dtype='int32'),
            '500100':       np.array([5], dtype='int32')
        },

        'face': {
            'angular':      np.array([0], dtype='int32'),
            'brick':        np.array([1], dtype='int32'),
            'circle':       np.array([2], dtype='int32'),
            'oval':         np.array([3], dtype='int32')
        },

        'mouth': {
            'medium':       np.array([0], dtype="int32"),
            'thick':        np.array([1], dtype="int32"),
            'thin':         np.array([2], dtype="int32")
        },

        'face_exp': {
            'kiss':         np.array([0], dtype="int32"),
            'open':         np.array([1], dtype="int32"),
            'open-smile':   np.array([2], dtype="int32"),
            'smile':        np.array([3], dtype="int32")
        },

        'brows': {
            'dooga-thin':   np.array([0], dtype="int32"),
            'dooga-thick':  np.array([1], dtype="int32"),
            'galka-thin':   np.array([2], dtype="int32"),
            'galka-thick':  np.array([3], dtype="int32"),
            'straight-thin': np.array([4], dtype="int32"),
            'straight-thick': np.array([5], dtype="int32")
        },

        'nose': {
            'down':         np.array([0], dtype="int32"),
            'up':           np.array([1], dtype="int32"),
            'normal':       np.array([2], dtype="int32"),
            'black':        np.array([3], dtype="int32")
        }
    }
    return tasks

# TODO fill true tasks names in the order that they was presented in previous function
def get_tasks_names():
    tasks_names = [
        # 'skin',
        # 'gender',
        # 'hair_cover',
        # 'hair_color',
        # 'hair_len',
        # 'hair_type',
        # 'hair_fringe',
        # 'beard',
        # 'glasses',
        'face',
        'mouth',
        'nose',
        'face_exp',
        'brows'
    ]
    return tasks_names


def get_file_params(mode):
    assert mode in ['classification', 'landmarks','3D'], \
        'Mode {} must be one from list {}. Pls check mode param.'.format(mode, '[classification, landmarks, 3D]')

    file_params = None

    if mode == 'landmarks':
        file_params = {
                        'landmarks': {
                                        'csv_filename': csv_filename,
                                        'names':        None,
                                        'types':        None,
                                        'sep':          landmarks_sep
                                    }
                      }

    if mode == 'classification':
        file_params = {
                        'labels': {
                                        'csv_filename': csv_filename,
                                        'names':        labels_names,
                                        'types':        labels_types,
                                        'sep':          labels_sep
                                    },
                        'microclasses': {
                                        'csv_filename': microclasses_filename,
                                        'names':        microclasses_names,
                                        'types':        microclasses_types,
                                        'sep':          microclasses_sep
                                    },
                        'landmarks': {
                                        'csv_filename': 'landmarks.csv',
                                        'names':        landmarks_names,
                                        'types':        landmarks_types,
                                        'sep':          labels_sep
                                     }
                   }

    if mode == '3D':
        file_params = {
                    'csv_filename': None,
                    'names':        None,
                    'types':        None,
                    'sep':          None
                   }
    return file_params


def get_augmentation_params(mode):
    assert mode in ['classification', 'landmarks', '3D'], 'Mode {} must be one from list {}. Pls check mode param.'.format(mode,'[classification, landmarks, 3D]')

    params = None

    if mode == 'landmarks':
        params = {
                    'distortion': {
                                    'do':           False,
                                    'schemes':      None
                                  },
                    'rotation':   {
                                    'do':           True,
                                    'angles':       get_angles
                                  },
                    'mirror':     {
                                    'do':           True,
                                    'new_order':    get_new_order_119()
                                  }
                 }

    if mode == 'classification':
        params = {
                    'distortion': {
                                    'do':           False,
                                    'schemes':      None
                                  },
                    'rotation':   {
                                    'do':           True,
                                    'angles':       get_angles
                                  },
                    'mirror':     {
                                    'do':           True,
                                    'new_order':    get_new_order_119()
                                  }
                 }

    if mode == '3D':
        params = {
                    'distortion': {
                                    'do':           False,
                                    'schemes':      None
                                  },
                    'rotation':   {
                                    'do':           True,
                                    'angles':       get_angles
                                  },
                    'mirror':     {
                                    'do':           True,
                                    'new_order':    None
                                  }
                 }
    return params


def get_angles(mode):
    assert mode in ['classification', 'landmarks', '3D'], 'Mode {} must be one from list {}. Pls check mode param.'.format(mode,'[classification, landmarks, 3D]')

    if mode == 'landmarks':
        return get_angles_landmarks
    if mode == 'classification':
        return get_angles_classification
    if mode == '3D':
        return get_angles_3D

def get_angles_landmarks(dirpath):
    return angles

def get_angles_classification(dirpath):
    import fnmatch

    cnt = len(fnmatch.filter(os.listdir(dirpath), '*.jpg'))
    if cnt > 200:
        return [3]
    else:
        return [6]

# TO DO: define
def get_angles_3D(dirpath):
    return angles
