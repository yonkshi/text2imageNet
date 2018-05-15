#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: MNIST.py
# Author: Qian Ge <geqian1001@gmail.com>

import os

import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

from ..base import RNGDataFlow

__all__ = ['MNIST', 'MNISTLabel']

def get_mnist_im_label(name, mnist_data):
    if name == 'train':
        return mnist_data.train.images, mnist_data.train.labels
    elif name == 'val':
        return mnist_data.validation.images, mnist_data.validation.labels
    else:
        return mnist_data.test.images, mnist_data.test.labels

# TODO read data without tensorflow
class MNIST(RNGDataFlow):
    """

    """
    def __init__(self, name, data_dir='', shuffle=True, normalize=None):

        self.num_channels = 1
        self.im_size = [28, 28]

        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        self.shuffle = shuffle
        self._normalize = normalize

        assert name in ['train', 'test', 'val']
        self.setup(epoch_val=0, batch_size=1)

        self._load_files(name)
        self._num_image = self.size()
        self._image_id = 0
        
    def _load_files(self, name):
        mnist_data = input_data.read_data_sets(self.data_dir, one_hot=False)
        self.im_list = []
        self.label_list = []

        mnist_images, mnist_labels = get_mnist_im_label(name, mnist_data)
        for image, label in zip(mnist_images, mnist_labels):
            # TODO to be modified
            if self._normalize == 'tanh':
                image = image*2.-1.
            image = np.reshape(image, [28, 28, 1])
            self.im_list.append(image)
            self.label_list.append(label)
        self.im_list = np.array(self.im_list)
        self.label_list = np.array(self.label_list)

        if self.shuffle:
            self._suffle_files()

    def _suffle_files(self):
        idxs = np.arange(self.size())

        self.rng.shuffle(idxs)
        self.im_list = self.im_list[idxs]
        self.label_list = self.label_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_files = self.im_list[start:end]

        if self._image_id + self._batch_size > self._num_image:
            self._epochs_completed += 1
            self._image_id = 0
            if self.shuffle:
                self._suffle_files()
        return [batch_files]

class MNISTLabel(MNIST):

    def next_batch(self):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_im = self.im_list[start:end]
        batch_label = self.label_list[start:end]

        if self._image_id + self._batch_size > self._num_image:
            self._epochs_completed += 1
            self._image_id = 0
            if self.shuffle:
                self._suffle_files()
        return [batch_im, batch_label]
   

if __name__ == '__main__':
    a = MNISTLabel('val','D:\\Qian\\GitHub\\workspace\\tensorflow-DCGAN\\MNIST_data\\')
    t = a.next_batch()
    print(t)
