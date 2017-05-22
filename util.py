#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import os
import time
import numpy as np
import shutil
import sys
from PIL import Image
from os import stat
from pwd import getpwuid


def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)


def labels_array_to_list(labels):
    assert isinstance(labels, (np.ndarray, np.generic)) , 'Labels must be type of \'numpy ndarray\'. Pls check type of input.'
    return [list([labels[2 * k], labels[2 * k + 1]]) for k in range(len(labels) / 2)]


def labels_str_to_list(labels):
    print type(labels)
    assert isinstance(labels, str) , 'Labels must be type of \'str\'. Pls check type of input.'
    l = labels.split(', ')
    return [map(float, list(l[i][1:-1].split(' '))) for i in range (len(l))]


def labels_str_to_flatten_list(labels):
    #print type(labels)
    assert isinstance(labels, str) , 'Labels must be type of \'str\'. Pls check type of input.'
    l = labels.split(', ')
    list_of_lists = [map(float, l[i][1:-1].split(' ')) for i in range(len(l))]
    flattened = [val for sublist in list_of_lists for val in sublist]
    return np.array(flattened)


def copy_nomatched_file(path_scr_dir, path_dst_dir, nomatch_filename):
    for root, subFolders, files in os.walk(path_scr_dir):
        for f in files:
            if f != nomatch_filename:
                path_src = os.path.join(path_scr_dir, f)
                path_dst = os.path.join(path_dst_dir, f)
                shutil.copy2(path_src, path_dst)


def remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def clean_csv(filename):
    f = open(filename, "w")
    f.truncate()
    f.close()


def print_cnt(cnt, whole, ratio, start):
    if cnt % ratio == 0:
        string = str(cnt + 1) + ' / ' + str(whole)
        sys.stdout.write("\r%s" % string)
        sys.stdout.write("\r{}. Elapsed time: {:.3f} sec".format(string, time.time() - start))
        sys.stdout.flush()


def create_file_with_paths_to_images(source_dir, path_to_file, path_to_labels):
    labels = np.load(path_to_labels)
    img_cnt, _ = labels.shape
    text_file = open(path_to_file, "w")
    for i in range(img_cnt):
        text_file.write("%s\n" % (os.path.join(source_dir, str(i).zfill(8) + '.jpg')))
    text_file.close()

def get_value(names, tasks_names, task):
    return names[tasks_names.index(task)]

def get_inode(dirpath):
    return int(str(os.stat(dirpath).st_ino)[-3:])

def is_empty_file(path_to_file):
    return os.stat(path_to_file).st_size == 0

def get_image_size(x):
    im = Image.open(x)
    return im.size[0]

def recompute_row(row):
    vals = np.array(row[:-1]).astype(float)
    size = float(row[-1])
    return np.append(vals/(size/2.) - 1, [size])