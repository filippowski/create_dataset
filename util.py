#!/usr/bin/env python2
## -*- coding: utf-8 -*-
import json
import os
import time
import numpy as np
import shutil
import sys
import math
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


def copy_auxiliary_files(path_scr_dir, path_dst_dir, needless_files_names=(), extensions=()):
    for root, subFolders, files in os.walk(path_scr_dir):
        for f in files:
            if f not in needless_files_names and os.path.splitext(f)[1] in extensions:
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

def points_as_array(points):
    num = len([p.x for p in points.parts()])
    points_as_array = np.zeros(2 * num, dtype=np.float32)
    idx = 0
    for point in points.parts():
        points_as_array[2 * idx] = point.x
        points_as_array[2 * idx + 1] = point.y
        idx += 1
    return points_as_array


def get_dist(left, top, right, bottom):
    return math.sqrt((right - left) ** 2 + (bottom - top) ** 2)

def get_alphas_from_alphasfile(path_to_alphas, alphas_num):
    alphas = np.zeros((alphas_num), dtype='float64')
    alphasfile = open(path_to_alphas)
    for line in alphasfile:
        alphas = np.array(line[1:-1].split(', '), dtype='float64')
    return alphas

def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_mark(alpha):
    mark = 0
    if alpha < -54:
        mark = 0
    elif alpha >= -54 and alpha < 54: 
        mark = 1
    elif alpha >= 54:
        mark = 2
    return mark


def get_points_from_json(json_path):
    d = json_load(json_path)
    pts =   d["points"]
    #pts = [item for sublist in pts for item in sublist]
    label = np.asarray(pts, dtype='float64')
    return label


def put_points_in_json(json_path, pts):
    dictionary = json_load(json_path)
    print dictionary["points"]
    dictionary["points"] = pts
    print dictionary["points"]
    with open(json_path, "w") as json_file:
        json.dump(dictionary, json_file)


# TODO refactor it
def get_labels_from_json(json_path):
    d = json_load(json_path)
    alpha = d["alpha"]
    #alpha = alpha[0]
    #mark = get_mark(alpha)
    #betta = d["betta"]
    pts =   d["points"]
    #pts = [item for sublist in pts for item in sublist]
    #label = np.asarray(([alpha]+[mark]+pts), dtype='float64')
    label = np.asarray((alpha+pts), dtype='float64')
    return label


def rewrite_points_in_json(json_path, ):
    d = json_load(json_path)
    alpha = d["alpha"]
    #alpha = alpha[0]
    #mark = get_mark(alpha)
    #betta = d["betta"]
    pts =   d["points"]
    pts = [item for sublist in pts for item in sublist]
    #label = np.asarray(([alpha]+[mark]+pts), dtype='float64')
    label = np.asarray((alpha+pts), dtype='float64')
    return label


def points_2Dlist_to_array(points):
    pts = [item for sublist in points for item in sublist]
    pts_as_array = np.array(pts).astype(float)
    return pts_as_array