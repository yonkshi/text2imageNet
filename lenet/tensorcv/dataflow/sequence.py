#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sequence.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import collections

from .base import DataFlow
from .normalization import identity
from ..utils.utils import assert_type

class SeqDataflow(DataFlow):
    """ base class for sequence data
     
    """
    def __init__(self, data_dir='',
                 load_ratio=1,
                 predict_step=0,
                 batch_dict_name=None,
                 normalize_fnc=identity):
        self._pred_step = predict_step
        self._data_dir = data_dir
        self._normalize_fnc = normalize_fnc
        self._load_ratio = load_ratio

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self.load_entire_seq()

        self._data_id = 0

        self.setup(epoch_val=0, batch_size=1)
        self.setup_seq_para(num_step=10, stride=1)
        self._updata_batch_partition_len()

    def _updata_batch_partition_len(self):
        try:
            self._batch_partition_len = self.size() // self._batch_size
        except AttributeError:
            pass

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        self._updata_batch_partition_len()

    def size(self):
        return len(self.get_entire_seq())

    def setup_seq_para(self, num_step, stride):
        self._num_step = num_step
        self._stride = stride

    def next_batch(self):
        b_size = self._batch_size
        bp_len = self._batch_partition_len
        assert b_size * self._num_step <= self.size()
        if self._data_id + bp_len * (b_size - 1) + self._num_step + self._pred_step > self.size():
            self._epochs_completed += 1
            # self._data_id  = 0
            # self._data_id = self._epochs_completed
            self._data_id = self._epochs_completed % self._num_step
            # self._data_id = self._epochs_completed % (bp_len - self._num_step - self._pred_step)
        start_id = self._data_id

        batch_data = []
        for i in range(b_size):
            start_id = self._data_id + bp_len * i
            end_id = start_id + self._num_step
            cur_data = self.load_data(start_id, end_id)
            batch_data.append(cur_data)

        self._data_id += self._num_step
        # self._data_id += 
        # print(np.array(batch_data).shape)
        return self._batch_transform(batch_data)

    def _batch_transform(self, batch_data):
        return batch_data

        # if len(np.array(batch_data).shape) == 3:
        #     return np.array(batch_data).transpose(1, 0, 2)
        # else:
        #     return np.array(batch_data).transpose(1, 0, 2, 3)

    # def next_batch(self):
    #     assert self.size() > self._batch_size * self._stride + self._num_step - self._stride
    #     batch_data = []
    #     batch_id = 0
    #     start_id = self._data_id
    #     while batch_id < self._batch_size:
    #         end_id = start_id + self._num_step
    #         if end_id + 1 > self.size():
    #             start_id = 0
    #             end_id = start_id + self._num_step
    #             self._epochs_completed += 1
    #         cur_data = self.load_data(start_id, end_id)
    #         batch_data.append(cur_data)
    #         start_id = start_id + self._stride
    #         batch_id += 1
    #     return np.array(batch_data).transpose(1, 0, 2)

    def load_data(self, start_id, end_id):
        pass
        # return self.get_entire_seq()[start_id: end_id]

    def load_entire_seq(self):
        pass

    def get_entire_seq(self):
        pass


class SepWord(SeqDataflow):
    def __init__(self, data_dir='',
                 predict_step=1,
                 word_dict=None,
                 batch_dict_name=None,
                 normalize_fnc=identity):
        self.word_dict = word_dict
        super(SepWord, self).__init__(data_dir=data_dir,
                                      predict_step=predict_step,
                                      batch_dict_name=batch_dict_name,
                                      normalize_fnc=normalize_fnc)

    def gen_word_dict(self, word_data):
        counter = collections.Counter(word_data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        self.word_dict = dict(zip(words, range(len(words))))


class SeqNumber(SeqDataflow):
    def _scale(self, data):
        normal_dict = self._normalize_fnc(data)
        try:
            self.scale_dict = normal_dict['scale_dict'] 
        except KeyError:
            pass
        return normal_dict['data']

        # max_data = np.amax(data)
        # min_data = np.amin(data)
        # return (data - min_data) / (max_data - min_data)

    def load_data(self, start_id, end_id):
        feature_seq = self.get_entire_seq()[start_id: end_id]
        label = self.get_label_seq()[start_id + self._pred_step: end_id + self._pred_step]
        return [feature_seq, label]

    def load_entire_seq(self):
        pass

    def get_entire_seq(self):
        pass

    def get_label_seq(self):
        pass
