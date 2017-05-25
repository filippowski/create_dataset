#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, csv
from scipy.ndimage import imread
from skimage.io import imsave
import multiprocessing as mp
from  multiprocessing import cpu_count, Pool
from util import ensure_dir, copy_nomatched_file
from load import load_cls_labels, load_cls_microclasses

# CREATE microclasses.csv
class Microclasses:

    def __init__(self, path_to_superdir, file_params, tasks_names, tasks):
        self.path_to_superdir = path_to_superdir
        assert os.path.exists(self.path_to_superdir), 'Path to superdir {} does not exist. Pls check path.'.format(self.path_to_superdir)

        self.tasks_names = tasks_names
        self.tasks       = tasks

        self.num_microclasses           = self.get_num_microclasses(self.tasks, self.tasks_names[1])
        self.num_nonempty_microclasses  = None

        self.labels_filename        = file_params['labels']['csv_filename']
        self.labels_names           = file_params['labels']['names']
        self.labels_types           = file_params['labels']['types']
        self.labels_sep             = file_params['labels']['sep']

        self.microclasses_filename  = file_params['microclasses']['csv_filename']
        self.microclasses_names     = file_params['microclasses']['names']
        self.microclasses_types     = file_params['microclasses']['types']
        self.microclasses_sep       = file_params['microclasses']['sep']

        self.path_to_labels         = os.path.join(self.path_to_superdir, self.labels_filename)
        self.path_to_microclasses   = os.path.join(self.path_to_superdir, self.microclasses_filename)

        assert os.path.exists(self.path_to_labels), \
            'Path to labels {} does not exist. Pls check path.'.format(self.path_to_labels)

    def get_num_microclasses(self, tasks, tasks_names):
        cnt = 1
        for el in tasks_names:
            cnt *= len(tasks[el].keys())
        return cnt

    # Create new table with counts of microclasses elements
    def write_microclasses_csv(self, path_to_microclasses, microclasses_sep, tasks_names, tasks, dataset, num_microclasses):
        print tasks_names
        # first task
        task = tasks_names[0]
        print "{}. Fill table for task \"{}\"".format(1, task)
        df = pd.DataFrame(index=range(0, num_microclasses), columns=[task])
        lst = [x for x in tasks_names if x != task]
        num_iter = self.get_num_microclasses(tasks, lst)
        keys = tasks[task].keys()
        #print num_iter, keys
        for cls in range(len(keys)):
            df[cls * num_iter:(cls + 1) * num_iter] = keys[cls]
        # rest tasks
        n_task = 1
        for task in tasks_names[1:]:
            n_task +=1
            print "{}. Fill table for task \"{}\"".format(n_task, task)
            df1 = pd.DataFrame(index=range(0, num_microclasses), columns=[task])
            lst_before = tasks_names[:tasks_names.index(task)]
            lst_after = tasks_names[tasks_names.index(task):]
            num_iter_global = self.get_num_microclasses(tasks, lst_before)
            num_iter_local  = self.get_num_microclasses(tasks, lst_after)
            keys = tasks[task].keys()
            num_iter = num_iter_local / len(keys)
            for i in range(num_iter_global):
                for cls in range(len(keys)):
                    df1[(i * num_iter_local) + cls * num_iter:(i * num_iter_local) + (cls + 1) * num_iter] = keys[cls]
            df = pd.concat([df, df1], axis=1)
        df.to_csv(path_to_microclasses, sep=microclasses_sep, mode='w+', header=False)
        print "Done: initial table was filled and saved."

    def isnan(self, value):
        try:
            import math
            return math.isnan(float(value))
        except:
            return False

    def create_microclasses_csv(self):
        # read initial csv with labels
        dataset_full = load_cls_labels(self.path_to_labels, self.labels_sep, self.tasks_names[0], self.labels_names, self.labels_types)
        dataset = dataset_full.iloc[:, 1:]
        print ' * dataset_full shape is: ',     dataset_full.shape
        print ' * dataset_full zero row is: ',  dataset_full.iloc[0]
        print ' * dataset shape is: ',          dataset.shape
        print ' * dataset_zero row is: ',       dataset.iloc[0]

        # create new table with counts of microclasses elements
        self.write_microclasses_csv(self.path_to_microclasses, self.microclasses_sep, self.tasks_names[1], self.tasks, dataset, self.num_microclasses)

        # samples from microclasses names and microclasses types for only those are in tasks_names
        microclasses_names = [x for x in self.microclasses_names if x in self.tasks_names[1]]
        microclasses_types = {key: self.microclasses_types[key] for key in self.tasks_names[1]}

        # read initial csv with microclasses
        all_microclasses = pd.read_csv(self.path_to_microclasses, sep=self.microclasses_sep, header=None, names=microclasses_names, dtype=microclasses_types)

        # add new columns to table
        count_ = pd.DataFrame(index=range(0, self.num_microclasses), columns = ['count'])
        filenames_list = pd.DataFrame(index=range(0, self.num_microclasses), columns = ['filenames_list'])
        # extended table
        all_microclasses = pd.concat([all_microclasses, count_, filenames_list], axis = 1).copy()
        print ' * microclasses.shape: ', all_microclasses.shape
        print all_microclasses

        # fill new cols with values
        for idx, row in dataset.iterrows():
            filename = dataset_full.iloc[idx, 0]
            #print ' * processing file: {}'.format(filename)

            df = all_microclasses
            index = 0
            for key in [x for x in row.keys() if x in self.tasks_names[1]]:
                if key in self.tasks_names[1]:
                    index = df[df[key] == row[key]].index.tolist()[0]
                    df = df[df[key] == row[key]]

            #print 'before: ', all_microclasses.iloc[index]

            # renew or write value in the table
            if self.isnan(all_microclasses.iloc[index]['count']):
                all_microclasses.iloc[index]['count'] = 1
            else:
                all_microclasses.iloc[index]['count'] = int(all_microclasses.iloc[index]['count']) + 1

            if self.isnan(all_microclasses.iloc[index]['filenames_list']):
                all_microclasses.iloc[index]['filenames_list'] = filename
            else:
                all_microclasses.iloc[index]['filenames_list'] = all_microclasses.iloc[index]['filenames_list'] + ' ' + filename

            #print 'after: ', all_microclasses.iloc[index]
        print ' * microclasses.shape: ', all_microclasses.shape
        print all_microclasses

        nonempty_microclasses = all_microclasses[all_microclasses['count'].notnull()]
        self.num_nonempty_microclasses = nonempty_microclasses.shape[0]
        print 'Number of nonempty microclasses is {}.'.format(self.num_nonempty_microclasses)

        nonempty_microclasses.to_csv(self.path_to_microclasses, sep=self.microclasses_sep, mode='w', header=False, index=False)