#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import config as cfg
from util import Profiler, create_file_with_paths_to_images
from check import Check
import warnings
warnings.filterwarnings("ignore")

#######################################################################################################################

# CREATE DATASET
def create_dataset():

    file_params = cfg.get_file_params(cfg.mode)
    crop_params = cfg.get_crop_params(cfg.mode)
    task_params = cfg.get_task_params(cfg.mode)
    augm_params = cfg.get_augm_params(cfg.mode)
    merg_params = cfg.get_merg_params(cfg.mode)
    lmdb_params = cfg.get_lmdb_params(cfg.mode)

    # Checking
    begin = True
    if cfg.check:
        check = Check(cfg.path_to_superdir, file_params, cfg.mode)
        check.run()
        begin = check.res

    if (begin):

        print '\n\nBeginning...\n\n'
        with Profiler() as p:

            # augmentation dataset
            if cfg.augmentation:
                with Profiler() as p:
                    from augmentation import Augmentation
                    aug = Augmentation(cfg.path_to_superdir, file_params, augm_params, task_params, cfg.mode)
                    aug.run_augmentation()

            # merge dataset
            if cfg.merge:
                with Profiler() as p:
                    from merge import Merge
                    mrg = Merge(cfg.path_to_superdir, cfg.main_path, file_params, crop_params, merg_params, task_params, cfg.mode)
                    mrg.merge()

            # create lmdb
            if cfg.create_lmdb:
                with Profiler() as p:
                    from create_lmdb import Lmdb
                    lmdb = Lmdb(cfg.main_path,
                                cfg.images_filename,
                                cfg.labels_filename,
                                cfg.path_to_lmdb_with_images,
                                cfg.path_to_lmdb_with_labels,
                                lmdb_params)
                    lmdb.create_lmdb()


#######################################################################################################################
if __name__ == '__main__':
    create_dataset()