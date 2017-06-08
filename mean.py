import os
import time
import numpy as np
from caffe.io import array_to_blobproto
from skimage import io

# CREATE MEAN IMAGE
class MeanImage:

    def __init__(self, main_path, path_to_file_with_paths_to_images, meanPrefix, crop_params):

        self.main_path = main_path
        self.path_to_file_with_paths_to_images = path_to_file_with_paths_to_images
        assert os.path.exists(self.main_path), \
            'Path to superdir {} does not exist. Pls check path.'.format(self.main_path)
        assert os.path.exists(self.path_to_file_with_paths_to_images), \
            'Path to file with paths to images {} does not exist. Pls check path.'.format(self.path_to_file_with_paths_to_images)

        self.imgSize    = crop_params['imgSize']
        self.channel    = crop_params['channel']
        self.img_ext    = crop_params['img_ext']
        self.meanPrefix = meanPrefix
        self.N          = 0
        self.start      = time.time()
        if self.channel == 3:
            self.mean = np.zeros((self.imgSize, self.imgSize, 3))
        elif self.channel == 1:
            self.mean = np.zeros((self.imgSize, self.imgSize))


    def create_mean_image(self):

        print '\n\n * creating meanimage\n'

        flname = os.path.join(self.main_path, self.meanPrefix)

        with open(self.path_to_file_with_paths_to_images, 'r') as f:
            paths = f.readlines()

        for path in paths:
            img = io.imread(path[:-1])
            self.update_mean(img)

        mean = self.mean / self.N

        # save mean as npy file
        self.save_mean_npy(flname, mean)
        # save mean as png file
        self.save_mean_png(flname, mean)
        # save mean as binaryproto file
        self.save_mean_binaryproto(flname, mean)

        print '\n/************************************************************************/'
        print 'Done: mean image created.'


    def update_mean(self, img):
        # channel = 3
        if img.shape == (self.imgSize, self.imgSize, 3):
            self.mean[:, :, 0] += img[:, :, 0]
            self.mean[:, :, 1] += img[:, :, 1]
            self.mean[:, :, 2] += img[:, :, 2]
        # channel = 1
        if img.shape == (self.imgSize, self.imgSize):
            self.mean += img[:, :]

        self.N += 1
        if self.N % 1000 == 0:
            elapsed = time.time() - self.start
            print("Processed {} images in {:.2f} seconds. "
                  "{:.2f} images/second.".format(self.N, elapsed,
                                                 self.N / elapsed))


    def save_mean_npy(self, flname, mean):
        np.save("{}.npy".format(flname), mean)


    def save_mean_png(self, flname, mean):
        # Images of type float must be between -1 and 1.
        mean_img = (mean - mean.min()) / float(mean.max() - mean.min())
        io.imsave("{}.png".format(flname), mean_img)  # img / 256.


    def save_mean_binaryproto(self, flname, mean):
        meanImg = mean_img
        if self.channel == 3:
            meanImg = np.transpose(mean.astype(np.uint8), (2, 0, 1))
            meanImg = meanImg.reshape((1, meanImg.shape[0], meanImg.shape[1], meanImg.shape[2]))
        elif self.channel == 1:
            meanImg = mean.astype(np.uint8)
            meanImg = meanImg.reshape((1, 1, meanImg.shape[0], meanImg.shape[1]))

        blob = array_to_blobproto(meanImg)
        with open("{}.binaryproto".format(flname), 'wb') as f:
            f.write(blob.SerializeToString())


    # TO DO refactoring
    def create_mean_image_by_task(self, main_path, path_to_superdir, csv_filename, directory_with_images, imgSize, channel):
        meanPrefix = 'mean_'
        flname = os.path.join(main_path, meanPrefix)
        imageDir = os.path.join(main_path, directory_with_images)
        exts = ["jpg", "png"]

        # csv filename and path-to-files defs
        path_to_initial_csv_file = os.path.join(path_to_superdir, csv_filename)

        # read initial csv 'labels.csv'
        dataset = pd.read_csv(path_to_initial_csv_file, sep=cfg.labels_sep, header=None,
                              names=cfg.labels_names,
                              dtype=cfg.labels_types)

        dataset['glasses'] = dataset['glasses'].replace('200200', '200100')
        tasks_names = get_tasks_names()
        tasks = get_tasks_digits()

        for task in tasks_names:
            for key in tasks[task].keys():
                dataset_task = dataset[dataset[task] == key].reset_index(drop=False)
                print task, key, dataset_task.shape

                if channel == 3:
                    mean = np.zeros((imgSize, imgSize, 3))
                elif channel == 1:
                    mean = np.zeros((imgSize, imgSize))
                N = 0

                beginTime = time.time()

                for idx, row in dataset_task.iterrows():
                    filename = '{}.jpg'.format(str(dataset_task.iloc[idx]['index']).zfill(8))
                    img = io.imread(os.path.join(imageDir, filename))

                    if img.shape == (imgSize, imgSize, 3):
                        mean[:, :, 0] += img[:, :, 0]
                        mean[:, :, 1] += img[:, :, 1]
                        mean[:, :, 2] += img[:, :, 2]
                        N += 1
                        if N % 1000 == 0:
                            elapsed = time.time() - beginTime
                            print("Processed {} images in {:.2f} seconds. "
                                  "{:.2f} images/second.".format(N, elapsed, N / elapsed))

                    if img.shape == (imgSize, imgSize):
                        mean += img[:, :]
                        N += 1
                        if N % 1000 == 0:
                            elapsed = time.time() - beginTime
                            print("Processed {} images in {:.2f} seconds. "
                                  "{:.2f} images/second.".format(N, elapsed, N / elapsed))

                print N
                mean /= float(N)

                # Images of type float must be between -1 and 1.
                mean_img = (mean - mean.min()) / float(mean.max() - mean.min())
                io.imsave("{}.png".format(flname + task + '_' + key), mean_img)  # img / 256.

        print '\n/************************************************************************/'
        print '\nDone: mean images per tasks created.'


    def create_infogain_matrices(main_path, path_to_superdir, csv_filename):
        import caffe

        print '\nCreating infogain matrices.\n'

        prefix = 'infogain_matrix_'
        flname = os.path.join(main_path, prefix)

        # csv filename and path-to-files defs
        path_to_initial_csv_file = os.path.join(path_to_superdir, csv_filename)

        # read initial csv 'labels.csv'
        dataset = pd.read_csv(path_to_initial_csv_file, sep=cfg.labels_sep, header=None,
                              names=cfg.labels_names,
                              dtype=cfg.labels_types)

        dataset['glasses'] = dataset['glasses'].replace('200200', '200100')
        tasks_names = get_tasks_names()
        tasks = get_tasks_digits()

        # create infogain matrices
        #
        # infogain matrix H is identity matrix of dim = number of classes in the task
        #
        # H(i, j) = 0                if i != j
        # H(i, j) = mean/cnt(i)      if i == j    (where cnt(i) = the count of class i examples in the dataset
        #                                                mean   = the mean count examples per class)
        # H(i, j) = 1 - cnt(i)/gen_cnt

        for task in tasks_names:
            groupby_task = dataset.groupby(task).size()
            mean = groupby_task.mean()
            gencnt = groupby_task.sum()
            L = len(groupby_task)
            H = np.eye(L, dtype='f4')
            for key in tasks[task].keys():
                cnt = groupby_task[key]
                idx = tasks[task][key][0]
                # H[idx, idx] = mean / float(cnt) #work slow
                # H[idx, idx] = 1 -float(cnt)/gencnt #work slow
                # H[idx,idx] = 1 - float(cnt-mean)/mean #work bad
                H[idx, idx] = mean / float(cnt)

            sm = 0
            for i in range(L):
                sm += H[i, i]
            for i in range(L):
                H[i, i] = H[i, i] * L / sm

            blob = caffe.io.array_to_blobproto(H.reshape((1, 1, L, L)))
            with open("{}.binaryproto".format(flname + task), 'wb') as f:
                f.write(blob.SerializeToString())

        print '\n/************************************************************************/'
        print '\nDone: infogain matrices created.'