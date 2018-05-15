# File: matlab.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
from scipy.io import loadmat

import numpy as np 

from .base import RNGDataFlow
from .common import *

__all__ = ['MatlabData']

class MatlabData(RNGDataFlow):
    """ dataflow from .mat file with mask """
    def __init__(self, 
                 data_dir='',
                 mat_name_list=None, 
                 mat_type_list=None,
                 shuffle=True,
                 normalize=None):

        self.setup(epoch_val=0, batch_size=1)

        self.shuffle = shuffle
        self._normalize = normalize

        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        assert mat_name_list is not None, 'mat_name_list cannot be None'
        if not isinstance(mat_name_list, list):
            mat_name_list = [mat_name_list]
        self._mat_name_list = mat_name_list
        if mat_type_list is None:
            mat_type_list = ['float']*len(self._mat_name_list)
        assert len(self._mat_name_list) == len(mat_type_list),\
        'Length of mat_name_list and mat_type_list has to be the same!'
        self._mat_type_list = mat_type_list

        self._load_file_list()
        self._get_im_size()
        self._num_image = self.size()
        self._image_id = 0

    def _get_im_size(self):
        # Run after _load_file_list
        # Assume all the image have the same size
        mat = loadmat(self.file_list[0])
        cur_mat = load_image_from_mat(mat, self._mat_name_list[0], 
                                      self._mat_type_list[0])
        if len(cur_mat.shape) < 3:
            self.num_channels = 1
        else:
            self.num_channels = cur_mat.shape[2]
        self.im_size = [cur_mat.shape[0], cur_mat.shape[1]]
        
    def _load_file_list(self):
        # data_dir = os.path.join(self.data_dir)
        self.file_list = np.array([os.path.join(self.data_dir, file) 
            for file in os.listdir(self.data_dir) if file.endswith(".mat")])

        if self.shuffle:
            self._suffle_file_list()

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_file_path = self.file_list[start:end]

        if self._image_id + self._batch_size > self._num_image:
            self._epochs_completed += 1
            self._image_id = 0
            if self.shuffle:
                self._suffle_file_list()
        return self._load_data(batch_file_path)

    def _load_data(self, batch_file_path):
        # TODO deal with num_channels
        input_data = [[] for i in range(len(self._mat_name_list))]

        for file_path in batch_file_path:
            mat = loadmat(file_path)
            cur_data = load_image_from_mat(mat, self._mat_name_list[0], 
                                      self._mat_type_list[0])
            cur_data = np.reshape(cur_data, 
                [1, cur_data.shape[0], cur_data.shape[1], self.num_channels])
            input_data[0].extend(cur_data)

            for k in range(1, len(self._mat_name_list)):
                cur_data = load_image_from_mat(mat, 
                               self._mat_name_list[k], self._mat_type_list[k])
                cur_data = np.reshape(cur_data, 
                               [1, cur_data.shape[0], cur_data.shape[1]])
                input_data[k].extend(cur_data)
        input_data = [np.array(data) for data in input_data]

        if self._normalize == 'tanh':
            try:
                input_data[0] = tanh_normalization(input_data[0], self._half_in_val)
            except AttributeError:
                self._input_val_range(input_data[0][0])
                input_data[0] = tanh_normalization(input_data[0], self._half_in_val)

        return input_data

    def _input_val_range(self, in_mat):
        # TODO to be modified  
        self._max_in_val, self._half_in_val = input_val_range(in_mat)  

    def size(self):
        return len(self.file_list)

def load_image_from_mat(matfile, name, datatype):
    mat = matfile[name].astype(datatype)
    return mat

if __name__ == '__main__':
    a = MatlabData(data_dir='D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_GAN_ORIGINAL_64\\', 
                   mat_name_list=['level1Edge'],
                   normalize='tanh')
    print(a.next_batch()[0].shape)
    print(a.next_batch()[0][:,30:40,30:40,:])
    print(np.amax(a.next_batch()[0]))