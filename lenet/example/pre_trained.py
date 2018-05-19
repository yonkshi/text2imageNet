#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pre_trained.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import numpy as np
import scipy.misc
import argparse

from lenet.tensorcv.dataflow.image import ImageFromFile

import lenet.example.setup_env as conf
from lenet.lib.nets.googlenet import GoogleNet
from lenet.lib.utils.preprocess import resize_image_with_smallest_side, center_crop_image
from lenet.lib.utils.classes import get_word_list
import matplotlib.pyplot as plt


def display_data(dataflow, data_name):
    try:
        print('[{}] num of samples {}, num of classes {}'.
              format(data_name, dataflow.size(), len(dataflow.label_dict)))
    except AttributeError:
        print('[{}] num of samples {}'.
              format(data_name, dataflow.size()))
    print(dataflow._im_list)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default='.jpg', type=str,
                        help='image file extension')
    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()

    model = GoogleNet(is_load=True, pre_train_path=conf.PARA_DIR)

    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    test_data = ImageFromFile(FLAGS.type,
                              data_dir=conf.DATA_DIR,
                              num_channel=3)
    #display_data(test_data, 'test_data')

    word_dict = get_word_list('data/imageNetLabel.txt')

    model.create_model([image, 1])
    test_op = tf.nn.top_k(tf.nn.softmax(model.layer['output']),
                          k=5, sorted=True)
    embed_op = model.layer['embedding']
    input_op = model.layer['input']

    writer = tf.summary.FileWriter(conf.SAVE_DIR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for k in range(0, 50):
            if test_data.epochs_completed < 1:
                batch_data = test_data.next_batch()
                im2 = batch_data[0]

                im = resize_image_with_smallest_side(im2, 224)

                # im = center_crop_image(im, 224, 224)
                # scipy.misc.imsave('{}test_{}.png'.format(conf.SAVE_DIR, k),
                #                   np.squeeze(im))
                encoded, output = sess.run([embed_op, test_op], feed_dict={image: im})
                for val, ind in zip(output.values, output.indices):
                    print('===========')
                    print(['%0.2f%%  ' % v for v in val])
                    print(word_dict[ind[0]])
