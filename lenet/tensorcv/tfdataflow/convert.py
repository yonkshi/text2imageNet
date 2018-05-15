#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: convert.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from ..utils.utils import assert_type 
from ..dataflow.base import DataFlow


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_image(image):
    return bytes_feature(tf.compat.as_bytes(image.tostring()))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dataflow2tfrecord(dataflow, tfname, record_names, c_fncs):
    assert_type(dataflow, DataFlow)
    dataflow.setup(epoch_val=0, batch_size=1)

    if not isinstance(record_names, list):
        record_names = [record_names]
    if not isinstance(c_fncs, list):
        c_fncs = [c_fncs]
    assert len(c_fncs) == len(record_names)

    tfrecords_filename = tfname
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    while dataflow.epochs_completed < 1:
        batch_data = dataflow.next_batch()
        feature = {}
        for record_name, convert_fnc, data in\
        	zip(record_names, c_fncs, batch_data):
            feature[record_name] = convert_fnc(data[0])

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
