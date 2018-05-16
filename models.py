import tensorflow as tf
from tensorflow.contrib import rnn
# import keras
import numpy as np

def build_char_cnn_rnn(input_seqs):
    with tf.variable_scope("txt_encode"):
        cnn_dim = 256 # dont know?
        embed_dim = 1024
        alphasize = 70 # like in paper?

        # 201 x alphasize
        conv1 = tf.layers.conv1d(input_seqs, 384, 4, activation=tf.nn.relu, name='txt_conv1')
        max1 = tf.layers.max_pooling1d(conv1, pool_size=[3,3], name='txt_max1')

        # 66 x 384
        conv2 = tf.layers.conv1d(max1, 512, 4, activation=tf.nn.relu, name='txt_conv2')
        max2 = tf.layers.max_pooling1d(conv2, pool_size=[3,3], name='txt_max2')

        # 21 x 512
        conv3 = tf.layers.conv1d(max2, cnn_dim, 4, activation=tf.nn.relu, name='txt_conv3')
        cnn_out = tf.layers.max_pooling1d(conv3, pool_size=[3,2], name='txt_max3')

        # 8 x cnn_dim
        rnn_cell = rnn.BasicRNNCell(cnn_dim, activation='relu', name='txt_rnn_cell') # 1 hidden layer
        outputs, states = rnn.static_rnn(rnn_cell, cnn_out, dtype=tf.float32)

        normalized = tf.truediv(tf.reduce_sum(outputs, axis=0),8)

        out = tf.layers.Dense(normalized, embed_dim, name='txt_upscale_dense')

    return out



