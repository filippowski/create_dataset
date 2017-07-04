from PIL import Image
import numpy as np
import lmdb
import caffe
import sys
import os
import math
import random


class Lmdb:

    def __init__(self,  main_path,
                        images_filename,
                        labels_filename,
                        path_to_lmdb_with_images,
                        path_to_lmdb_with_labels,
                        lmdb_params):

        self.main_path  = main_path
        self.images     = os.path.join(main_path, images_filename)
        self.labels     = os.path.join(main_path, labels_filename)
        self.imagesOut  = path_to_lmdb_with_images
        self.labelsOut  = path_to_lmdb_with_labels
        self.test_data_percent = lmdb_params['testSize']
        self.maxPx      = lmdb_params['imgSize']
        self.minPx      = lmdb_params['imgSize']
        self.shuffle    = lmdb_params['shuffle']
        self.ndim       = lmdb_params['channel']
        self.lmdb_mode  = lmdb_params['lmdb_mode']
        self.imgs_cnt   = None
        self.lbls_cnt   = None
        self.size_one_img = None
        self.size_one_lbl = None

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

        images_map_size = 3 * len(images) * self.size_one_img       #self.maxPx * self.minPx * self.ndim
        labels_map_size = 20 * len(images) * self.size_one_lbl       #self.lbls_cnt
        print '\nImages map size: ', images_map_size
        print 'Labels map size: ', labels_map_size

        images_db = lmdb.open(images_file, map_size=images_map_size, map_async=True, writemap=True)
        labels_db = lmdb.open(labels_file, map_size=labels_map_size, map_async=True, writemap=True)

        #print 'size imgdb: ', os.stat(os.path.join(images_file,'data.mdb')).st_size
        #print 'size lbldb: ', os.stat(os.path.join(labels_file,'data.mdb')).st_size

        images_txn = images_db.begin(write=True)
        labels_txn = labels_db.begin(write=True)

        examples = zip(images, labels)
        for in_idx, (image, label) in enumerate(examples):
            try:
                # write image to lmdb
                im = Image.open(os.path.join(self.main_path, image))
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
                string_ = str(in_idx + 1) + ' / ' + str(len(images))
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

        images_map_size = 2 * len(images) * self.size_one_img       #self.maxPx * self.minPx * self.ndim
        labels_map_size = 3 * len(images) * self.size_one_lbl       #self.lbls_cnt
        print 'Images map size: ', images_map_size
        print 'Labels map size: ', labels_map_size

        images_db = lmdb.open(images_file, map_size=images_map_size, map_async=True, writemap=True)
        images_txn = images_db.begin(write=True)

        labels_db, labels_txn = [], []

        for i in range(self.lbls_cnt):
            labels_db.append(lmdb.open(labels_file + '_' + str(i), map_size=labels_map_size))
            labels_txn.append(labels_db[i].begin(write=True))
            print i, type(labels_db), type(labels_txn), type(labels_db[i]), type(labels_txn[i])
        print len(labels_db), len(labels_txn)

        examples = zip(images, labels)
        for in_idx, (image, label) in enumerate(examples):
            try:
                # write image to lmdb
                path2im = os.path.join(self.main_path, image)
                print "************"
                im = Image.open(path2im)
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
                    lbl = np.array(label[i]).astype(float).reshape(1, 1, 1)
                    label_dat = caffe.io.array_to_datum(np.array([]), lbl)

                    # put label in label-LMDB (as new pair <key, value>)
                    labels_txn[i].put('{:0>8d}'.format(in_idx) + '_0', label_dat.SerializeToString())

                cnt += 1

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print e
                print "Skipped image and label with id {0}".format(in_idx)
            if in_idx % 1000 == 0:
                string_ = str(in_idx + 1) + ' / ' + str(len(images))
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

        print '\n\n * creating LMDBs\n'

        fillLmdb = self.fillLmdb if self.lmdb_mode == 'caffe' else self.fillLmdb_one_lmdb_per_one_label

        # images
        images = np.loadtxt(self.images, str, delimiter='\t')
        images = np.array([self.main_path+im for im in images])

        self.imgs_cnt = len(images)
        im = np.array(Image.open(images[0]))
        self.size_one_img = sys.getsizeof(im)
        print "\nNumber of images: {}".format(self.imgs_cnt)
        print "Size of one image: {} Bytes".format(self.size_one_img)
        # labels
        labels = np.load(self.labels)
        self.lbls_cnt = labels.shape[1]
        lbl = np.array(labels[0]).astype(float).reshape(1, 1, labels[0].shape[0])
        self.size_one_lbl = sys.getsizeof(lbl)
        print "\nNumber of labels: {}".format(self.lbls_cnt)
        print "Size of one label: {} Bytes".format(self.size_one_lbl)


        num = self.test_data_percent * self.imgs_cnt / 100

        if self.shuffle:
            print "\n * shuffling the data"
            data = zip(images, labels)
            random.shuffle(data)
            images, labels = zip(*data)
            images = np.array(images)
            #labels = np.array(labels)


        if num != 0:
            print "\n * creating test set"
            fillLmdb(
                images_file=self.imagesOut + "_test",
                labels_file=self.labelsOut + "_test",
                ndim=self.ndim,
                images=images[:num],
                labels=labels[:num])

        print "\n * creating training set"
        fillLmdb(
            images_file=self.imagesOut,
            labels_file=self.labelsOut,
            ndim=self.ndim,
            images=images[num:],
            labels=labels[num:])

        print self.labels
        print '\n/************************************************************************/'
        print 'Done: dataset successfully created.\n\n'
