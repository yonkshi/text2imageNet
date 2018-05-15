#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from ..dataflow.base import DataFlow
from ..dataflow.normalization import identity
from ..utils.utils import assert_type


class DataFromTfrecord(DataFlow):
    def __init__(self, tfname,
                 record_names,
                 record_types,
                 raw_types,
                 decode_fncs,
                 batch_dict_name,
                 shuffle=True,
                 data_shape=[],
                 feature_len_list=None,
                 pf=identity):

        if not isinstance(tfname, list):
            tfname = [tfname]
        if not isinstance(record_names, list):
            record_names = [record_names]
        if not isinstance(record_types, list):
            record_types = [record_types]
        for c_type in record_types:
            assert_type(c_type, tf.DType)
        if not isinstance(raw_types, list):
            raw_types = [raw_types]
        for raw_type in raw_types:
            assert_type(raw_type, tf.DType)
        if not isinstance(decode_fncs, list):
            decode_fncs = [decode_fncs]
        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]   
        assert len(record_types) == len(record_names)
        assert len(record_types) == len(batch_dict_name)

        if feature_len_list is None:
            feature_len_list = [[] for i in range(0, len(record_names))]
        elif not isinstance(feature_len_list, list):
            feature_len_list = [feature_len_list]
        self._feat_len_list = feature_len_list
        if len(self._feat_len_list) < len(record_names):
            self._feat_len_list.extend([[] for i in range(0, len(record_names) - len(self._feat_len_list))])

        self.record_names = record_names
        self.record_types = record_types
        self.raw_types = raw_types
        self.decode_fncs = decode_fncs
        self.data_shape = data_shape
        self._tfname = tfname
        self._batch_dict_name = batch_dict_name

        self._shuffle = shuffle

        # self._batch_step = 0
        # self.reset_epochs_completed(0)
        # self.set_batch_size(batch_size)
        self.setup_decode_data()
        self.setup(epoch_val=0, batch_size=1)


    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        self.updata_data_op(batch_size)
        self.updata_step_per_epoch(batch_size)

    def updata_data_op(self, batch_size):
        try:
            if self._shuffle is True:
                self._data = tf.train.shuffle_batch(
                    self._decode_data,
                    batch_size=batch_size,
                    capacity=batch_size * 4,
                    num_threads=2,
                    min_after_dequeue=batch_size * 2)
            else:
                print('***** data is not shuffled *****')
                self._data = tf.train.batch(
                    self._decode_data,
                    batch_size=batch_size,
                    capacity=batch_size,
                    num_threads=1,
                    allow_smaller_final_batch=False)
            # self._data = self._decode_data[0]
            # print(self._data)
        except AttributeError:
            pass

    def reset_epochs_completed(self, val):
        self._epochs_completed  = val
        self._batch_step = 0

    # def _setup(self, **kwargs):
    def setup_decode_data(self):    
        # n_epoch = kwargs['num_epoch']
        
        feature = {}
        for record_name, r_type, cur_size in zip(self.record_names, self.record_types, self._feat_len_list):
            feature[record_name] = tf.FixedLenFeature(cur_size, r_type)
        # filename_queue = tf.train.string_input_producer(self._tfname, num_epochs=n_epoch)
        filename_queue = tf.train.string_input_producer(self._tfname)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        decode_data = [decode_fnc(features[record_name], raw_type)
                       for decode_fnc, record_name, raw_type
                       in zip(self.decode_fncs, self.record_names, self.raw_types)]

        for idx, c_shape in enumerate(self.data_shape):
            if c_shape:
                decode_data[idx] = tf.reshape(decode_data[idx], c_shape)

        self._decode_data = decode_data

        # self._data = self._decode_data[0]
        try:
            self.set_batch_size(batch_size=self._batch_size)
        except AttributeError:
            self.set_batch_size(batch_size=1)

    def updata_step_per_epoch(self, batch_size):
        self._step_per_epoch = int(self.size() / batch_size)

    def before_read_setup(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def next_batch(self):
        sess = tf.get_default_session()
        batch_data = sess.run(self._data)
        self._batch_step += 1
        if self._batch_step % self._step_per_epoch == 0:
            self._epochs_completed += 1
        # print(batch_data[2])
        return batch_data

    def next_batch_dict(self):
        sess = tf.get_default_session()
        batch_data = sess.run(self._data)
        self._batch_step += 1
        if self._batch_step % self._step_per_epoch == 0:
            self._epochs_completed += 1
        batch_dict = {name: data for name, data in zip(self._batch_dict_name, batch_data)}
        return batch_dict

    def after_reading(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

    def size(self):
        try:
            return self._size
        except AttributeError:
            self._size = sum(1 for f in self._tfname for _ in tf.python_io.tf_record_iterator(f))
            return self._size
        