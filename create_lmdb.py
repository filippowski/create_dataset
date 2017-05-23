from PIL import Image
import numpy as np
import lmdb
import caffe
import sys
import math
import random


class Lmdb:

    def __init__(self,  path_to_file_with_paths_to_images,
                        path_to_labels,
                        path_to_lmdb_with_images,
                        path_to_lmdb_with_labels,
                        testSize=20,
                        imgSize=224,
                        channel=3,
                        shuffle=True):

        self.images     = path_to_file_with_paths_to_images
        self.labels     = path_to_labels
        self.ndim       = channel
        self.imagesOut  = path_to_lmdb_with_images
        self.labelsOut  = path_to_lmdb_with_labels
        self.test_data_persent = testSize
        self.maxPx      = imgSize
        self.minPx      = imgSize
        self.shuffle    = shuffle
        self.imgs_cnt   = None
        self.lbls_cnt   = None

        assert os.path.exists(self.images), \
            'Path to images txt file {} does not exist. Pls check path.'.format(self.images)
        assert os.path.exists(self.labels), \
            'Path to labels npy file {} does not exist. Pls check path.'.format(self.labels)

    def resize(self, img):
        try:
            width = img.size[0]
            height = img.size[1]
            smallest = min(width, height)
            largest = max(width, height)
            k = 1
            if largest > self.maxPx:
                k = self.maxPx / float(largest)
                smallest *= k
                largest *= k
            if smallest < self.minPx:
                k *= self.minPx / float(smallest)
            size = int(math.ceil(width * k)), int(math.ceil(height * k))
            img = img.resize(size, Image.ANTIALIAS)
            return img
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)


    def fillLmdb(self, images_file, labels_file, ndim, images, labels):
        means = np.zeros(ndim)
        cnt = 0
        # print images.shape[0]
        images_map_size = 2 * self.imgs_cnt * self.maxPx * self.minPx * self.ndim       # 5 * 10e4 * self.img_cnt
        labels_map_size = 1 * self.imgs_cnt                                             # 5 * 10e3

        images_db = lmdb.open(images_file, map_size=images_map_size, map_async=True, writemap=True)
        labels_db = lmdb.open(labels_file, map_size=labels_map_size)

        images_txn = images_db.begin(write=True)
        labels_txn = labels_db.begin(write=True)

        examples = zip(images, labels)
        for in_idx, (image, label) in enumerate(examples):
            try:
                # write image to lmdb
                im = Image.open(image)
                im = np.array(self.resize(im))
                # renew mean
                mean = im.mean(axis=0).mean(axis=0)
                means += mean

                if ndim == 3:
                    im = im[:, :, ::-1]
                    im = im.transpose((2, 0, 1))
                    im_dat = caffe.io.array_to_datum(im)
                elif ndim == 1:
                    im = np.array([im])
                    im_dat = caffe.io.array_to_datum(im)

                # put image in image-LMDB (as new pair <key, value>)
                images_txn.put('{:0>8d}'.format(in_idx) + '_0', im_dat.SerializeToString())

                # write label to lmdb
                label = np.array(label).astype(float).reshape(1, 1, label.shape[0])
                label_dat = caffe.io.array_to_datum(label)

                # put label in label-LMDB (as new pair <key, value>)
                labels_txn.put('{:0>8d}'.format(in_idx) + '_0', label_dat.SerializeToString())

                cnt += 1

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print e
                print "Skipped image and label with id {0}".format(in_idx)
            if in_idx % 1000 == 0:
                string_ = str(in_idx + 1) + ' / ' + str(self.imgs_cnt)
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()

        images_txn.commit()
        labels_txn.commit()
        images_db.close()
        labels_db.close()

        print "\nFilling lmdb completed"
        print "Image mean values for RBG: {0}".format(means / cnt)


    def fillLmdb_one_lmdb_per_one_label(self, images_file, labels_file, ndim, images, labels):
        means = np.zeros(ndim)
        cnt = 0
        # print images.shape[0]
        images_map_size = 2 * self.imgs_cnt * self.maxPx * self.minPx * self.ndim       # 5 * 10e4 * self.img_cnt
        labels_map_size = 1 * self.imgs_cnt                                             # 5 * 10e3

        images_db = lmdb.open(images_file, map_size=images_map_size, map_async=True, writemap=True)
        images_txn = images_db.begin(write=True)

        labels_db, labels_txn = [], []

        for i in range(self.lbls_cnt):
            labels_db.append(lmdb.open(labels_file + '_' + str(i), map_size=labels_map_size))
            labels_txn.append(labels_db.begin(write=True))

        examples = zip(images, labels)
        for in_idx, (image, label) in enumerate(examples):
            try:
                # write image to lmdb
                im = Image.open(image)
                im = np.array(self.resize(im))
                # renew mean
                mean = im.mean(axis=0).mean(axis=0)
                means += mean

                if ndim == 3:
                    im = im[:, :, ::-1]
                    im = im.transpose((2, 0, 1))
                    im_dat = caffe.io.array_to_datum(im)
                elif ndim == 1:
                    im = np.array([im])
                    im_dat = caffe.io.array_to_datum(im)

                # put image in image-LMDB (as new pair <key, value>)
                images_txn.put('{:0>8d}'.format(in_idx) + '_0', im_dat.SerializeToString())

                for i in range(self.lbls_cnt):
                    # write label to lmdb
                    label = np.array(label[i]).astype(float).reshape(1, 1, label.shape[0])
                    label_dat = caffe.io.array_to_datum(label)

                    # put label in label-LMDB (as new pair <key, value>)
                    labels_txn[i].put('{:0>8d}'.format(in_idx) + '_0', label_dat.SerializeToString())

                cnt += 1

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print e
                print "Skipped image and label with id {0}".format(in_idx)
            if in_idx % 1000 == 0:
                string_ = str(in_idx + 1) + ' / ' + str(self.imgs_cnt)
                sys.stdout.write("\r%s" % string_)
                sys.stdout.flush()

        images_txn.commit()
        images_db.close()

        for i in range(self.lbls_cnt):
            labels_txn[i].commit()
            labels_db[i].close()

        print "\nFilling lmdb completed"
        print "Image mean values for RBG: {0}".format(means / cnt)


    def create_lmdb(self):

        images = np.loadtxt(self.images, str, delimiter='\t')
        self.imgs_cnt = len(images)
        print "Number of images: {}".format(self.imgs_cnt)
        labels = np.load(self.labels)
        self.lbls_cnt = labels.shape[1]
        num = self.test_data_persent * self.img_cnt / 100

        if self.shuffle:
            print "Shuffling the data"
            data = zip(images, labels)
            random.shuffle(data)
            images, labels = zip(*data)
            images = np.array(images)
            labels = np.array(labels)

        if num != 0:
            print "Creating test set"
            fillLmdb(
                images_file=self.imagesOut + "_test",
                labels_file=self.labelsOut + "_test",
                ndim=self.ndim,
                images=images[:num],
                labels=labels[:num],
                minPx=self.minPx,
                maxPx=self.maxPx)

        print "Creating training set"
        fillLmdb(
            images_file=self.imagesOut,
            labels_file=self.labelsOut,
            ndim=self.ndim,
            images=images[num:],
            labels=labels[num:],
            minPx=self.minPx,
            maxPx=self.maxPx)

        print '\n\nDone: dataset successfully created.'
