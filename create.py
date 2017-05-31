#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import config as cfg
from util import Profiler, create_file_with_paths_to_images
from check import Check
from config import get_tasks, get_tasks_names
import warnings
warnings.filterwarnings("ignore")

#######################################################################################################################

# CREATE DATASET
def create_dataset():

    # Checking
    print '\n\nChecking...\n\n'
    check = Check(cfg.path_to_superdir, cfg.mode)
    check.run()

    if (check.res):
    #if True:

        print '\n\nBeginning...\n\n'
        with Profiler() as p:

            file_params = cfg.get_file_params(cfg.mode)
            crop_params = cfg.get_crop_params(cfg.mode)
            task_params = cfg.get_task_params(cfg.mode)
            augm_params = cfg.get_augmentation_params(cfg.mode)
            merg_params = cfg.get_merge_params(cfg.mode)
            lmdb_params = cfg.get_lmdb_params(cfg.mode)

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

                # if merge_params['merge']:
                #     with Profiler() as p:
                #         from merge import Merge
                #         mrg = Merge(cfg.path_to_superdir, cfg.main_path, cfg.directory_with_images, file_params, crop_params, tasks_names, cfg.mode)
                #         mrg.merge()
                #
                # # create and save labels
                # if merge_params['create_labels']:
                #     with Profiler() as p:
                #         from labels import Label
                #         ## labels as numbers
                #         lbl = Label(cfg.main_path, file_params, cfg.path_to_labels, cfg.mode, cfg.task_mask, tasks, tasks_names)
                #         lbl.create_labels()
                #
                #         # labels as images
                #         #create_labels_imgs(main_path, csv_file_name, path_to_labels_imgs)
                #         #create_file_with_paths_to_images(path_to_labels_imgs, path_to_file_with_paths_to_labels_imgs, img_cnt)
                #
                # # create file with images filenames
                # if merge_params['create_imgfile']:
                #     with Profiler() as p:
                #         create_file_with_paths_to_images(cfg.path_to_dir_with_images, cfg.path_to_file_with_paths_to_images, cfg.path_to_labels)
                #
                # # create mean image
                # if merge_params['create_mean']:
                #     with Profiler() as p:
                #         from mean import MeanImage
                #         mimg = MeanImage(cfg.main_path, cfg.directory_with_images, cfg.imgSize, cfg.channel, cfg.meanPrefix)
                #         mimg.create_mean_image()
                #
                #         #create_mean_image_by_task(cfg.main_path, cfg.path_to_superdir, cfg.csv_filename, cfg.directory_with_images, cfg.imgSize, cfg.channel)
                #
                # # create infogain matrices
                # if merge_params['create_infogain']:
                #     with Profiler() as p:
                #         create_infogain_matrices(cfg.main_path, cfg.path_to_superdir, cfg.csv_filename)

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