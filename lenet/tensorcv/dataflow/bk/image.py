import numpy as np 
import collections
import scipy.io
import scipy.misc
import os
import pickle
import random

import tensorflow as tf

class CIFAT10(object):
    def __init__(self, file_path):
        self._file_list = [os.path.join(file_path, 'data_batch_' + str(batch_id)) for batch_id in range(1,6)]
        self._num_channels = 3
        self._batch_id = 0
        self._batch_file_id = -1
        self._image = []
        self._epochs_completed = 0

    def next_batch_file(self):
        if self._batch_file_id >= len(self._file_list)-1:
            self._batch_file_id = 0
            self._epochs_completed += 1
        else:
            self._batch_file_id += 1
        self._image = unpickle(self._file_list[self._batch_file_id])
        random.shuffle(self._image)
        # scipy.misc.imsave('test.png', self.image[100])

    def next_batch(self, batch_size):
        batch_id_end = self._batch_id + batch_size
        if batch_id_end >= len(self._image):
            self.next_batch_file()
            self._batch_id = 0
            batch_id_end = batch_size
        batch_image = self._image[self._batch_id:batch_id_end]
        self._batch_id = batch_id_end
        return batch_image

    @property
    def epochs_completed(self):
        return self._epochs_completed


class TestImage(object):
    def __init__(self, test_file_path, patch_size, sample_mean = 0, num_channels = 1):
        self.patch_size = patch_size
        self.num_channels = num_channels
        self._sample_mean = sample_mean

        self._file_list = ([os.path.join(test_file_path, file) 
        for file in os.listdir(test_file_path)
        if file.endswith(".mat")])
        
        self._image_id = -1
        self._image = None
        self._cnt_row = -1

        self.cur_patch = np.empty(shape=[1, self.patch_size, self.patch_size, self.num_channels])

    def next_image(self):
        if self._image_id >= len(self._file_list) - 1:
            return None
        self._image_id += 1
        self._image = load_image(self._file_list[self._image_id], self.num_channels)

        self.im_rows, self.im_cols = self._image.shape[0], self._image.shape[1]
        self._cnt_row = 1

        half_patch_size = int(self.patch_size/2)
        self.row_id, self.col_id = half_patch_size - 1, half_patch_size - 2
        self.patch_row_start, self.patch_row_end = self.row_id - half_patch_size + 1, self.row_id + half_patch_size 
        self.patch_col_start, self.patch_col_end = self.col_id - half_patch_size + 1, self.col_id + half_patch_size

        for cur_channel in range(0, self.num_channels):
            self._image[:,:,:,cur_channel] -= self._sample_mean[cur_channel]

        return self._image

    def next_patch(self):
        if self._image is None:
            return None
        half_patch_size = int(self.patch_size/2)
        if self.col_id >= self.im_cols - half_patch_size - 1:
            self.row_id += 1
            self.patch_row_start += 1
            self.patch_row_end += 1
            print('Row: ' + str(self.row_id))
            self._cnt_row += 1
            self.col_id = half_patch_size - 1
            self.patch_col_start, self.patch_col_end = self.col_id - half_patch_size + 1, self.col_id + half_patch_size      
        else:
            self.col_id += 1
            self.patch_col_start += 1 
            self.patch_col_end += 1

        if self.row_id >= self.im_rows - half_patch_size:
            return None
        # print(self.patch_row_start,self.patch_row_end,self.patch_col_start,self.patch_col_end )
        if self.num_channels > 1:
            cnt_depth = 0
            for channel_id in range(0, self.num_channels):
                self.cur_patch[:,:,:,cnt_depth] = self._image[self.patch_row_start:self.patch_row_end + 1,
                                                  self.patch_col_start:self.patch_col_end + 1, channel_id].transpose()
                cnt_depth += 1
        else:
            self.cur_patch[:,:,:,:] = np.reshape(self._image[self.patch_row_start:self.patch_row_end + 1, 
                                                   self.patch_col_start:self.patch_col_end + 1].transpose(), 
                                                   [1, self.patch_size, self.patch_size, self.num_channels])
        return self.cur_patch

    def reset_cnt_row(self):
        self._cnt_row = 1
    def get_cnt_row(self):
        return self._cnt_row

class TrainData(object):
    def __init__(self, file_list, sample_mean = 0, num_channels = 1):
        self._num_channels = num_channels

        self._file_list = file_list
        self._num_image = len(self._file_list)

        self._image_id = 0
        self._image = []
        self._label = []
        self._mask = []
        self._epochs_completed = 0

        self._sample_mean = sample_mean

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def set_epochs_completed(self, value):
        self._epochs_completed = value

    @property
    def sample_mean(self):
        return self._sample_mean

    def next_image(self):
        if self._image_id >= self._num_image - 1:
            self._epochs_completed += 1
            self._image_id = 1
            perm = np.arange(self._num_image)
            np.random.shuffle(perm)
            self._file_list = self._file_list[perm]
        else:
            self._image_id += 1

        self._image, self._label, self._mask = load_training_image(self._file_list[self._image_id], num_channels = self._num_channels)
        for cur_channel in range(0, self._num_channels):
            self._image[:,:,:,cur_channel] -= self._sample_mean[cur_channel]
        return self._image, self._label, self._mask

    def next_batch(self, batch_size):
        if batch_size > self._num_image:
            return None
        start = self._image_id
        self._image_id += batch_size
        if self._image_id >= self._num_image:
            self._epochs_completed += 1
            perm = np.arange(self._num_image)
            np.random.shuffle(perm)
            self._file_list = self._file_list[perm]
            start = 0
            self._image_id = batch_size
        end = self._image_id
        batch_file_path = self._file_list[start:end]
        return load_batch_image(batch_file_path, num_channels = self._num_channels)



def prepare_data_set(file_path, valid_percentage, num_channels = 1, isSubstractMean = True):
    file_list = np.array([os.path.join(file_path, file) 
        for file in os.listdir(file_path)
        if file.endswith(".mat")])
    if isSubstractMean:
        sample_mean_value = average_train_data(file_list, num_channels)
    else:
        sample_mean_value = np.empty(shape=[num_channels], dtype = float)
    sample_mean = tf.Variable(sample_mean_value, name = 'sample_mean')
    num_image = len(file_list)
    num_valid = int(num_image*valid_percentage)
    num_train = num_image - num_valid
    train = TrainData(file_list[:num_train], sample_mean = sample_mean_value, num_channels = num_channels)
    validate = TrainData(file_list[num_train:], sample_mean = sample_mean_value, num_channels = num_channels)
    print('Number of training image: {}. Number of validation image: {}.'.format(num_train, num_valid))
    # print('Mean of training samples: {}'.format(sampel_mean))
    ds = collections.namedtuple('TrainData', ['train', 'validate'])
    return ds(train = train, validate = validate)

def load_batch_image(batch_file_path, num_channels = 1):
    image_list = []
    # image_list = np.empty(shape = [0, im_size, im_size, num_channels])
    for file_path in batch_file_path:
        image ,_ ,_ = load_training_image(file_path, num_channels = num_channels)
        image_list.extend(image)
        # image_list = np.append(image_list, image, axis=0)
    return np.array(image_list)


def load_training_image(file_path, num_channels = 1):
    # print('Loading training file ' + file_path)
    mat = scipy.io.loadmat(file_path)
    image = mat['level1Edge'].astype('float')
    label = mat['GT']
    mask = mat['Mask']
    image = np.reshape(image, [1, image.shape[0], image.shape[1], num_channels])
    label = np.reshape(label, [1, label.shape[0], label.shape[1]])
    mask = np.reshape(mask, [1, mask.shape[0], mask.shape[1]])
    # print('Load successfully.') 
    return image, label, mask

def load_image(test_file_path, num_channels):
    print('Loading test file ' + test_file_path + '...')
    mat = scipy.io.loadmat(test_file_path)
    image = mat['level1Edge'].astype('float')
    print('Load successfully.') 
    return np.array(np.reshape(image, [1, image.shape[0], image.shape[1], num_channels]))

def average_train_data(file_list, num_channels):
    mean_list = np.empty(shape=[num_channels], dtype = float)
    for cur_file_path in file_list:
        image, label, mask = load_training_image(cur_file_path, num_channels = num_channels)
        for cur_channel in range(0, num_channels):
            mean_list[cur_channel] += np.ma.masked_array(image[:,:,:, cur_channel], mask = mask).mean()
    return mean_list/len(file_list)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    image = dict[b'data']

    r = image[:,:32*32].reshape(-1,32,32)
    g = image[:,32*32: 2*32*32].reshape(-1,32,32)
    b = image[:,2*32*32:].reshape(-1,32,32)

    image = np.stack((r,g,b),axis=-1)
    return image

