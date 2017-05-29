#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import os
import random
from newOrder import get_new_order_119
from util import get_value, get_inode


################################################################################################
# 1. Main parameters
# TODO set main params
################################################################################################
mode        = 'classification'
lmdb_mode   = 'caffe'

assert mode in ['classification', 'landmarks', '3D'], \
    'Mode {} must be one from list {}. Pls check mode param.'.format(mode, '[classification, landmarks, 3D]')
assert lmdb_mode in ['caffe', 'caffe2'], \
    'LMDB mode {} must be one from list {}. Pls check mode param.'.format(lmdb_mode, '[caffe, caffe2]')

################################################################################################
# 2. Key paths and names
# TODO set path and names
################################################################################################

# Full path to directory where is 'superdir' folder that contain some count of folders with images and 'landmarks.csv' files
main_path   = '/8TB/DATASETS/multitask_2/cls_datasets/nose/nose_tip'

images_filename         = 'images.txt'
labels_filename         = 'labels.npy'
directory_with_images   = 'train'
meanPrefix              = 'mean'
lmdb_images_name        = 'lmdb_images'
lmdb_labels_name        = 'lmdb_labels'

csv_filename            = 'labels.csv' if mode == 'classification' else 'landmarks.csv'
superdir_name           = 'results' if mode == '3D' else 'superdir'

# classification task
microclasses_filename   = 'microclasses.csv'

# 3D task
bunch_fldname           = 'bunch'
alphas_fldname          = 'alphas'
alphas_ext              = '.alpha'

path_to_superdir = os.path.join(main_path, superdir_name)
path_to_labels = os.path.join(main_path, labels_filename)
path_to_file_with_paths_to_images = os.path.join(main_path, images_filename)
path_to_dir_with_images = os.path.join(main_path, directory_with_images)
path_to_lmdb_with_images = os.path.join(main_path, lmdb_images_name)
path_to_lmdb_with_labels = os.path.join(main_path, lmdb_labels_name)
path_to_alphas = os.path.join(main_path, alphas_fldname)

################################################################################################
# 3. Flags
# TODO set flags
################################################################################################
# 3.1 Flags for dataset creation
################################################

# augmentation dataset
augmentation    = True
# merge all csv in one
merge           = True
# create and save labels
create_labels   = True
# create file with images filenames
create_imgfile  = True
# create mean image
create_mean     = True
# create lmdb
create_lmdb     = True
# create infogain matrices
create_infogain = False

################################################
# 3.2 Other flags
################################################

# do shift during images cropping
do_shft                 = True
# write to labels their mask in multitask classification
task_mask               = False
# do shuffle before create lmdb
shuffle                 = True
# run augmentation main scheme
run_main_IF_SCHEME_AUG  = False

################################################################################################
# 4. Digits parameters
# TODO set digits params
################################################################################################

# channels number of images
channel  = 3
# width and height size of image (image must be same width and height size)
imgSize  = 224
# percentage of test examples from whole dataset
testSize = 10

################################################################################################
# 5. Files parameters
# TODO fill true files params <key, value> pairs
################################################################################################

def get_file_params(mode):

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
                        'labels':    {
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
                        '3D':       {
                                        'csv_filename': None,
                                        'names':        None,
                                        'types':        None,
                                        'sep':          None
                                    }
                      }
    return file_params


################################################################################################
# 6. Augmentation parameters
# TODO fill true augmentation params <key, value> pairs
################################################################################################

def get_augmentation_params(mode):

    aug_params = None

    if mode == 'landmarks':
        aug_params = {
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
        aug_params = {
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
        aug_params = {
                    'distortion': {
                                    'do':           False,
                                    'schemes':      None
                                  },
                    'rotation':   {
                                    'do':           False,
                                    'angles':       get_angles
                                  },
                    'mirror':     {
                                    'do':           False,
                                    'new_order':    None
                                  }
                 }
    return aug_params


################################################################################################
# 7. Angles for rotation
# TODO set tasks and tasks names
################################################################################################
# 7.1 Angles defaults
# TODO fill true default values for parameters
################################################

# angles for rotation
angles = [3, 6, 9, 12, 15, 18, 21]
#angles = range(1, 60, 3)

# count of images per microclass for which will be decided run augmentation or not:
# run augmentation,     if count > threshold
# not run,              if count <= threshold
threshold = 50 #35000

max_angle  = 20
#max_angle  = int(np.ceil(float(threshold)) - 1)

# rotate more than one time by one angle
replace = True

################################################
# 7.2 Functions for defining angles
# TODO define functions to get right angles for rotations
################################################

def get_angles(mode):

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
    if cnt > threshold:
        return []
    else:
        cnt_before, cnt_after = cnt, threshold
        tasks_names = get_tasks_names()
        np.random.seed(get_inode(dirpath))
        __, dirname = os.path.split(dirpath)
        names = dirname.split('_')
        idx, names = names[0], names[1:]

        # main IF-scheme of data augmentation
        # **************************************************************
        if run_main_IF_SCHEME_AUG:
            if       get_value(names, tasks_names[1], 'hair_fringe'  ) == 'close':

                        cnt_after = 2*threshold

            elif     get_value(names, tasks_names[1], 'hair_color')  == 'black' \
              and (get_value(names, tasks_names[1], 'hair_len'  )    == '5'
                or get_value(names, tasks_names[1], 'hair_len'  )    == '6'):

                        cnt_after = 1.5*threshold

            elif     get_value(names, tasks_names[1], 'hair_type' )  == 'curly' \
                or get_value(names, tasks_names[1], 'hair_color')    == 'carroty':

                        cnt_after = 1.5*threshold

            elif     get_value(names, tasks_names[1], 'hair_fringe') == 'open' \
                or get_value(names, tasks_names[1], 'hair_color' )   == 'black' \
                or get_value(names, tasks_names[1], 'hair_type'  )   == 'undefined':

                        cnt_after = 0.5*threshold
        # **************************************************************

        cnt_angls = int(np.ceil(0.5 * (float(cnt_after) / cnt_before)) - 1)
        angles = np.random.choice(max_angle, size=cnt_angls, replace=replace)
        #print ' * dirname: {}, cnt_before: {}, cnt_after: {}, cnt_angls: {}, angles: {}'.format(dirname, cnt_before, cnt_after, cnt_angls, angles)
        return angles

def get_angles_3D(dirpath):
    return []


################################################################################################
# 8. Names and types
# TODO set names and types
################################################################################################
# 8.1 Separators
########################################################################

# разделитель лейблов в landmarks.csv
landmarks_sep    = ' '
# разделитель лейблов в labels.csv
labels_sep       = ';'
# разделитель лейблов в microclasses.csv
microclasses_sep = ' '

########################################################################
# 8.2 Names and types dictionaries
########################################################################
# 8.2.1 Names and types for LABELS.CSV (CLS)
################################################

# columns names in labels.csv for cls task
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
                #'nose_type',
                'nose_tip',
                'face',
                'mouth',
                'nose',
                'face_exp',
                'brows'
                ]

# columns types in labels.csv for cls task
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
                 #'nose_type':    str,
                 'nose_tip':     str,
                 #'face':         str,
                 #'mouth':        str,
                 #'nose':         str,
                 #'face_exp':     str,
                 #'brows':        str
                }

################################################
# 8.2.2 Names and types for LANDMARKS.CSV (CLS)
################################################

# columns names in landmarks.csv for cls task
landmarks_names = [
                'FILENAME_JPG',
                'bbox',
                'facepoints'
                ]

# columns types in landmarks.csv for cls task
landmarks_types = {
                 'FILENAME_JPG': str,
                 'bbox':         str,
                 'facepoints':   str
                }

################################################
# 8.2.3 Names and types for MICROCLASSES.CSV (CLS)
################################################

# columns names in microclasses.csv for cls task
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
# columns types in microclasses.csv for cls task
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


# columns names in microclasses.csv for cls task
microclasses2_names = [
                      'face',
                      'mouth',
                      'nose',
                      'face_exp',
                      'brows',
                      'nose_type',
                      'nose_tip',
                      'count',
                      'filenames_list'
                     ]
# columns types in microclasses.csv for cls task
microclasses2_types = {
                        'face':             str,
                        'mouth':            str,
                        'nose':             str,
                        'face_exp':         str,
                        'brows':            str,
                        'nose_type':        str,
                        'nose_tip':         str,
                        'count':            int,
                        'filenames_list':   str
                     }

########################################################################
# 8.3 Microclasses_names and types choice
########################################################################

# choose params for reading microclasses.csv file (type 1 or 2)
microclasses_names = microclasses2_names
microclasses_types = microclasses2_types

################################################################################################
# 9. Tasks and tasks names
# TODO set tasks and tasks names
################################################################################################
# 9.1 Tasks dictionary
# TODO fill true tasks <key, value> pairs
################################################

def get_tasks():
    tasks = {
        'skin': {
            'white':        np.array([0], dtype='int32'),
            'dark':         np.array([1], dtype='int32'),
            'asian':        np.array([2], dtype='int32'),
            'tawny':        np.array([3], dtype='int32')
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
            '500100':       np.array([5], dtype='int32'),
            '200200':       np.array([6], dtype='int32')
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
        },

        'nose_type': {
            'MANUAL_down':   np.array([0], dtype="int32"),
            'MANUAL_up':     np.array([1], dtype="int32"),
            'MANUAL_normal': np.array([2], dtype="int32")
        },

        'nose_tip': {
            'MANUAL_blunt':  np.array([0], dtype="int32"),
            'MANUAL_sharp':  np.array([1], dtype="int32")
        }
    }
    return tasks

################################################
# 9.2 Tasks names list
# TODO fill true tasks names in the order that they was presented in LABELS.CSV
################################################

def get_tasks_names():
    tasks_names_full = [
        'skin',
        'gender',
        'hair_cover',
        'hair_color',
        'hair_len',
        'hair_type',
        'hair_fringe',
        'beard',
        'glasses',
        #'face',
        #'mouth',
        #'nose',
        #'face_exp',
        #'brows',
        #'nose_type',
        'nose_tip'
    ]
    tasks_names_work = [
        #'skin',
        #'gender',
        #'hair_cover',
        #'hair_color',
        #'hair_len',
        #'hair_type',
        #'hair_fringe',
        #'beard',
        #'glasses',
        #'face',
        #'mouth',
        #'nose',
        #'face_exp',
        #'brows',
        #'nose_type',
        'nose_tip'
    ]
    return (tasks_names_full, tasks_names_work)

################################################################################################
#                                           THE END