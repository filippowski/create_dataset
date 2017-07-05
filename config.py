#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import os
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
main_path   = '/8TB/OhDL/multitask/30.06.2017'

images_filename         = 'images.txt'
labels_filename         = 'labels.npy'
directory_with_images   = 'train'
meanPrefix              = 'mean'
lmdb_images_name        = 'lmdb_images'
lmdb_labels_name        = 'lmdb_labels'

img_ext                 = '.jpg'

csv_filename            = 'landmarks.csv' if mode == 'landmarks' else 'labels.csv'
superdir_name           = 'superdir'

# classification task
microclasses_filename   = 'microclasses.csv'

# 3D task
bunch_fldname           = 'bunch'
alphas_fldname          = 'alphas'
alphas_ext              = '.alpha'
expression              = 'expression'
expression_strength     = 'expression_strength'
crop_endswith           = '_crop'

full_path_to_dlib_model = '/8TB/vitalii/dlib64-1.dat'

path_to_superdir         = os.path.join(main_path, superdir_name)
path_to_lmdb_with_images = os.path.join(main_path, lmdb_images_name)
path_to_lmdb_with_labels = os.path.join(main_path, lmdb_labels_name)
path_to_alphas           = os.path.join(path_to_superdir, alphas_fldname)

################################################################################################
# 3. Flags
# TODO set flags
################################################################################################
# 3.1 Flags for dataset creation
################################################

# do dataset checking
check           = False#True

# dataset augmentation
# detailed settings are available in Part 6. Augmentation parameters
augmentation    = True#False

# merge all data, create labels and mean image
# detailed settings are available in Part 7. Merge parameters
merge           = False#True

# lmdb create
# detailed settings are available in Part 7. LMDB parameters
create_lmdb     = False#True

################################################
# 3.2 Other flags
################################################

# do shuffle before lmdb create
shuffle                 = True
# run augmentation by main scheme
run_main_IF_SCHEME_AUG  = True#False

################################################################################################
# 4. Digits parameters
# TODO set digits params
################################################################################################

# percentage of test examples from whole dataset
testSize = 10
# width and height size of image (image must be same width and height size)
imgSize  = 224
# channels number of images
channel  = 3

################################################################################################
# 5. Crop parameters
# TODO fill true crop params <key, value> pairs
################################################################################################

def get_crop_params(mode):

    crop_params = None

    if mode == 'landmarks':
        crop_params = {
                        'do_shft':   True,      # do shift during images cropping
                        'shft':      10,
                        'cntr_pt':   37,
                        'coef':      1.5,
                        'imgSize':   imgSize,
                        'channel':   channel,
                        'img_ext':   img_ext,
                        'left_x':    (0,1,2,3),
                        'right_x':   (13,14,15,16),
                        'top_y':     (17,18,19),
                        'bot_y':     (7,8,9)
                     }

    if mode == 'classification':
        crop_params = {
                        'do_shft':   True,
                        'shft':      10,
                        'cntr_pt':   37,
                        'coef':      1.5,
                        'imgSize':   imgSize,
                        'channel':   channel,
                        'img_ext':   img_ext,
                        'left_x':    (0,1,2,3),
                        'right_x':   (13,14,15,16),
                        'top_y':     (17,18,19),
                        'bot_y':     (7,8,9)
                     }

    if mode == '3D':
        crop_params = {
                        'do_shft':   False,
                        'shft':      10,
                        'cntr_pt':   17,
                        'coef':      1.05,
                        'imgSize':   imgSize,
                        'channel':   channel,
                        'img_ext':   img_ext,
                        'left_x':    (0,1,2,3),
                        'right_x':   (13,14,15,16),
                        'top_y':     (23,24,25,34,35,36),
                        'bot_y':     (7,8,9)
                     }
    return crop_params


################################################################################################
# 6. Files parameters
# TODO fill true files params <key, value> pairs
################################################################################################

def get_file_params(mode):

    file_params = None

    if mode == 'landmarks':
        file_params = {
                        'in':   {
                                    'landmarks': {
                                                    'csv_filename': csv_filename,
                                                    'names': None,
                                                    'types': None,
                                                    'sep': landmarks_sep
                                                 }
                                },

                        'out':  {
                                    'labels_filename':          labels_filename,
                                    'images_filename':          images_filename,
                                    'directory_with_images':    directory_with_images,
                                    'meanPrefix':               meanPrefix
                                }
                      }

    if mode == 'classification':
        file_params = {
                        'in':   {
                                    'labels':    {
                                                    'csv_filename': csv_filename,
                                                    'names':        get_labels_names_and_types(get_tasks_names()[1])[0]
                                                    'types':        get_labels_names_and_types(get_tasks_names()[1])[1]
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
                                },

                        'out':  {
                                    'labels_filename':          labels_filename,
                                    'images_filename':          images_filename,
                                    'directory_with_images':    directory_with_images,
                                    'meanPrefix':               meanPrefix
                                }
                   }

    if mode == '3D':
        file_params = {
                        'in':   {
                                    'alphas':    {
                                                    'path_to_alphas':   path_to_alphas,
                                                    'bunch_fldname':    bunch_fldname,
                                                    'alphas_fldname':   alphas_fldname,
                                                    'alphas_ext':       alphas_ext,
                                                    'alphas_cnt':       50
                                                 },

                                    'bettas':    {
                                                    'expression':           expression,
                                                    'expression_strength':  expression_strength,
                                                    'expressions_names':    get_expressions_names(),
                                                    'bettas_cnt':           17#len(get_expressions_names())
                                                 },

                                    'pts':       {
                                                    'pts_cnt':              64
                                                 },

                                    'dlib_model': {
                                                    'path_to_model':    full_path_to_dlib_model,
                                                    'crop_endswith':    crop_endswith,
                                                    'imgs_ext':         img_ext,
                                                    'imgs_cnt':         42
                                                  }
                                },

                        'out':  {
                                    'labels_filename':          labels_filename,
                                    'images_filename':          images_filename,
                                    'directory_with_images':    directory_with_images,
                                    'meanPrefix':               meanPrefix
                                }
                      }
    return file_params

################################################################################################
# 7. Augmentation parameters
# TODO fill true augmentation params <key, value> pairs
################################################################################################

def get_augm_params(mode):

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
# 8. Merge parameters
# TODO fill true merge params <key, value> pairs
################################################################################################

def get_merg_params(mode):

    mrg_params = None

    if mode == 'landmarks':
        mrg_params = {
                        'merge':           True,
                        'create_labels':   True,
                        'create_imgfile':  True,
                        'create_mean':     False,
                        'create_infogain': False
                     }

    if mode == 'classification':
        mrg_params = {
                        'merge':           True,
                        'create_labels':   True,
                        'create_imgfile':  True,
                        'create_mean':     True,
                        'create_infogain': False
                     }

    if mode == '3D':
        mrg_params = {
                        'merge':           True,
                        'create_labels':   False,
                        'create_imgfile':  False,
                        'create_mean':     False,
                        'create_infogain': False
                     }

    return mrg_params


################################################################################################
# 9. LMDB parameters
# TODO fill true LMDB params <key, value> pairs
################################################################################################

def get_lmdb_params(mode):

    lmdb_params = None

    if mode == 'landmarks':
        lmdb_params = {
                        'testSize':  testSize,
                        'imgSize':   imgSize,
                        'channel':   channel,
                        'lmdb_mode': lmdb_mode,
                        'shuffle':   True
                     }

    if mode == 'classification':
        lmdb_params = {
                        'testSize':  testSize,
                        'imgSize':   imgSize,
                        'channel':   channel,
                        'lmdb_mode': lmdb_mode,
                        'shuffle':   True
                     }

    if mode == '3D':
        lmdb_params = {
                        'testSize':  testSize,
                        'imgSize':   imgSize,
                        'channel':   channel,
                        'lmdb_mode': lmdb_mode,
                        'shuffle':   True
                     }
    return lmdb_params

################################################################################################
# 10. Angles for rotation
# TODO set tasks and tasks names
################################################################################################
# 10.1 Angles defaults
# TODO fill true default values for parameters
################################################

# angles for rotation
angles = [6]

#angles = [3, 6, 9, 12, 15, 18, 21]
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
# 10.2 Functions for defining angles
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
    return angles

################################################################################################
# 11. Tasks and tasks names
# TODO set tasks and tasks names
################################################################################################
# 11.1 Tasks names list
# TODO fill true tasks names in the order that they was presented in LABELS.CSV
################################################

def get_tasks_names():
    tasks_names_in_labels_file = [
        'skin',
        'gender',
        'hair_cover',
        'hair_color',
        'hair_len',
        'hair_type',
        'hair_fringe',
        'beard',
        'glasses',
        'eyes_color',
        'age',
        'face',
        'mouth',
        'mouth_exp',
        'brows',
        'nose',
        'nose_tip',
        'nose_width',
        'nose_wings'
    ]
    tasks_names_to_work = [
        'skin',
        'gender',
        'hair_cover',
        'hair_color',
        'hair_len',
        'hair_type',
        'hair_fringe',
        'beard',
        'glasses',
        'eyes_color',
        'age'
    ]
    return (tasks_names_in_labels_file, tasks_names_to_work)

################################################
# 11.2 Tasks dictionary
# TODO fill true tasks <key, value> pairs
################################################

def get_tasks():
    tasks = {
        'skin': {
            'white':            np.array([0], dtype='int32'),
            'dark':             np.array([1], dtype='int32'),
            'asian':            np.array([2], dtype='int32'),
            'tawny':            np.array([3], dtype='int32')
        },

        'gender': {
            'male':             np.array([0], dtype='int32'),
            'female':           np.array([1], dtype='int32')
        },

        'hair_cover': {
            'no':               np.array([0], dtype='int32'),
            'hijab':            np.array([1], dtype='int32'),
            'hat':              np.array([2], dtype='int32')
        },

        'hair_color': {
            'black':            np.array([0], dtype='int32'),
            'brown':            np.array([1], dtype='int32'),
            'light-brown':      np.array([2], dtype='int32'),
            'blond':            np.array([3], dtype='int32'),
            'carroty':          np.array([4], dtype='int32'),
            'grey':             np.array([5], dtype='int32'),
            'undefined':        np.array([6], dtype='int32')
        },

        'hair_len': {
            '1':                np.array([0], dtype='int32'),
            '2':                np.array([1], dtype='int32'),
            '3':                np.array([2], dtype='int32'),
            '4':                np.array([3], dtype='int32'),
            '5':                np.array([4], dtype='int32'),
            '6':                np.array([5], dtype='int32'),
            'undefined':        np.array([6], dtype='int32')
        },

        'hair_type': {
            'curly':            np.array([0], dtype='int32'),
            'straight':         np.array([1], dtype='int32'),
            'wavy':             np.array([2], dtype='int32'),
            'undefined':        np.array([3], dtype='int32')
        },

        'hair_fringe': {
            'close':            np.array([0], dtype='int32'),
            'open':             np.array([1], dtype='int32'),
            'partial':          np.array([2], dtype='int32'),
            'undefined':        np.array([3], dtype='int32')
        },

        'beard': {
            'without':          np.array([0], dtype='int32'),
            'bristle':          np.array([1], dtype='int32'),
            'mustache':         np.array([2], dtype='int32'),
            'circle_beard':     np.array([3], dtype='int32'),
            'full_beard':       np.array([4], dtype='int32'),
            'just_beard':       np.array([5], dtype='int32'),
            'balbo':            np.array([6], dtype='int32')
        },

        'glasses': {
            '100000':           np.array([0], dtype='int32'),
            '100200':           np.array([1], dtype='int32'),
            '200100':           np.array([2], dtype='int32'),
            '300100':           np.array([3], dtype='int32'),
            '400100':           np.array([4], dtype='int32'),
            '500100':           np.array([5], dtype='int32'),
            '200200':           np.array([6], dtype='int32'),
            'aviators':         np.array([7], dtype='int32')
        },

        'eyes_color': {
            'black':            np.array([0], dtype='int32'),
            'brown':            np.array([1], dtype='int32'),
            'grey':             np.array([2], dtype='int32'),
            'blue':             np.array([3], dtype='int32'),
            'green':            np.array([4], dtype='int32'),
            'undefined':        np.array([5], dtype='int32')
        },

        'age': {
            'CLEAR_ADULT':      np.array([0], dtype='int32'),
            'CLEAR_YOUNG':      np.array([1], dtype='int32'),
            'CLEAR_TEENAGERS':  np.array([2], dtype='int32'),
            'CLEAR_TEENAGERS\t': np.array([2], dtype='int32'),
            'CLEAR_OLD':        np.array([3], dtype='int32'),
            'CLEAR_CHILD':      np.array([4], dtype='int32')
        },

        'face': {
            'angular':          np.array([0], dtype='int32'),
            'brick':            np.array([1], dtype='int32'),
            'circle':           np.array([2], dtype='int32'),
            'oval':             np.array([3], dtype='int32')
        },

        'mouth': {
            'medium':           np.array([0], dtype="int32"),
            'thick':            np.array([1], dtype="int32"),
            'thin':             np.array([2], dtype="int32")
        },

        'mouth_exp': {
            'kiss':             np.array([0], dtype="int32"),
            'open':             np.array([1], dtype="int32"),
            'open-smile':       np.array([2], dtype="int32"),
            'smile':            np.array([3], dtype="int32")
        },

        'brows': {
            'dooga-thin':       np.array([0], dtype="int32"),
            'dooga-thick':      np.array([1], dtype="int32"),
            'galka-thin':       np.array([2], dtype="int32"),
            'galka-thick':      np.array([3], dtype="int32"),
            'straight-thin':    np.array([4], dtype="int32"),
            'straight-thick':   np.array([5], dtype="int32")
        },

        'nose_old': {
            'down':             np.array([0], dtype="int32"),
            'up':               np.array([1], dtype="int32"),
            'normal':           np.array([2], dtype="int32"),
            'black':            np.array([3], dtype="int32")
        },

        'nose': {
            'down':             np.array([0], dtype="int32"),
            'up':               np.array([1], dtype="int32"),
            'normal':           np.array([2], dtype="int32")
        },

        'nose_tip': {
            'blunt':            np.array([0], dtype="int32"),
            'sharp':            np.array([1], dtype="int32")
        },

        'nose_width': {
            'average':          np.array([0], dtype="int32"),
            'wide':             np.array([1], dtype="int32")
        },

        'nose_wings': {
            'invisible':        np.array([0], dtype="int32"),
            'visible':          np.array([1], dtype="int32")
        }

    }
    return tasks

################################################
# 11.3 Task parameters
# TODO fill true task params <key, value> pairs
################################################

# write to labels their mask in multitask classification
task_mask = False

def get_task_params(mode):

    task_params = {
                    'tasks':            None,
                    'tasks_names':      None,
                    'tasks_mask':       None
                  }

    if mode == 'classification':
        task_params = {
                        'tasks':            get_tasks(),
                        'tasks_names':      get_tasks_names(),
                        'tasks_mask':       task_mask
                     }

    return task_params

################################################################################################
# 12. Names and types
# TODO set names and types
################################################################################################
# 12.1 Separators
########################################################################

# разделитель лейблов в landmarks.csv
landmarks_sep    = ' '
# разделитель лейблов в labels.csv
labels_sep       = ';'
# разделитель лейблов в microclasses.csv
microclasses_sep = ' '

########################################################################
# 12.2 Names and types dictionaries
########################################################################
# 12.2.1 Names and types for LABELS.CSV (CLS)
################################################

# columns names in labels.csv for cls task
def get_labels_names_and_types(tasks_names):
    labels_names = ['FILENAME_JPG']
    labels_names.extend(tasks_names)

    labels_types = dict()
    [labels_types.update({x: str}) for x in labels_names]

    return labels_names, labels_types

################################################
# 12.2.2 Names and types for LANDMARKS.CSV (CLS)
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
    'bbox': str,
    'facepoints': str
}

################################################
# 12.2.3 Names and types for MICROCLASSES.CSV (CLS)
################################################

microclasses_names = get_tasks_names()[0]
microclasses_names.extend(['count','filenames_list'])

microclasses_types = {'count' : int}
[microclasses_types.update({x: str}) for x in microclasses_names if x != 'count']

################################################################################################
#                                           THE END
################################################################################################