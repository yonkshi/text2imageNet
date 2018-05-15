#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: write.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from ..dataflow.base import DataFlow
from ..models.base import BaseModel
from ..utils.utils import assert_type
from .convert import float_feature


class Bottleneck2TFrecord(object):
    def __init__(self, nets, record_feat_names,
                 feat_preprocess=tf.identity):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            assert_type(net, BaseModel)
        if not isinstance(record_feat_names, list):
            record_feat_names = [record_feat_names]
        assert len(nets) == len(record_feat_names)
        self._w_f_names = record_feat_names

        self._feat_ops = []
        self._feed_plh_keys = []
        self._net_input_dicts = []
        for net in nets:
            net.set_is_training(False)
            net.create_graph()
            self._net_input_dicts.append(net.input_dict)
            self._feed_plh_keys.append(net.prediction_plh_dict)
            self._feat_ops.append(feat_preprocess(net.layer['conv_out']))

    def write(self, tfname, dataflow,
              record_dataflow_keys=[], record_dataflow_names=[], c_fncs=[]):
        assert_type(dataflow, DataFlow)
        dataflow.setup(epoch_val=0, batch_size=1)

        if not isinstance(record_dataflow_names, list):
            record_dataflow_names = [record_dataflow_names]
        if not isinstance(c_fncs, list):
            c_fncs = [c_fncs]
        if not isinstance(record_dataflow_keys, list):
            record_dataflow_keys = [record_dataflow_keys]
        assert len(c_fncs) == len(record_dataflow_names)
        assert len(record_dataflow_keys) == len(record_dataflow_names)

        tfrecords_filename = tfname
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            dataflow.before_read_setup()
            cnt = 0
            while dataflow.epochs_completed < 1:
                print('Writing data {}...'.format(cnt))
                batch_data = dataflow.next_batch_dict()

                feats = []
                for feat_op, feed_plh_key, net_input_dict in zip(self._feat_ops, self._feed_plh_keys, self._net_input_dicts):
                    feed_dict = {net_input_dict[key]: batch_data[key]
                                 for key in feed_plh_key}
                    feats.append(sess.run(feat_op, feed_dict=feed_dict))

                # feature = {}
                # for record_name, convert_fnc, key in zip(record_dataflow_names, c_fncs, record_dataflow_keys):
                #     feature[record_name] = convert_fnc(batch_data[key][0])


                # for record_name, feat in zip(self._w_f_names, feats):
                #     feature[record_name] = float_feature(feat.reshape(-1).tolist())

                # feature_list = []
                batch_size = len(feats[0])
                for idx in range(0, batch_size):
                    feature = {}
                    for record_name, convert_fnc, key in zip(record_dataflow_names, c_fncs, record_dataflow_keys):
                        feature[record_name] = convert_fnc(
                            batch_data[key][idx])

                    for record_name, feat in zip(self._w_f_names, feats):
                        feature[record_name] =\
                            float_feature(feat[idx].reshape(-1).tolist())

                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

                cnt += 1
            dataflow.after_reading()

        writer.flush()
        writer.close()
