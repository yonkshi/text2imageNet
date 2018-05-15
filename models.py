import tensorflow as tf
# import keras
import numpy as np

def build_char_cnn_rnn():

    with tf.variable_scope("charcnnrnn"):
        # TODO Alphasize
        input = None # TODO remove me
        cnn_dim = None
        alphasize = []

        # 201 x alphasize
        conv1 = tf.layers.conv1d(input, 384, 4, activation=tf.nn.relu)
        max1 = tf.layers.max_pooling1d(conv1, pool_size=[3,3])

        # 66 x 258
        conv2 = tf.layers.conv1d(max1, 512, 4, activation=tf.nn.relu)
        max2 = tf.layers.max_pooling1d(conv2, pool_size=[3,3])

        # 21 x 256
        conv3 = tf.layers.conv1d(max2, cnn_dim, 4, activation=tf.nn.relu)
        max3 = tf.layers.max_pooling1d(conv3, pool_size=[3,2])
    pass