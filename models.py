import tensorflow as tf
from tensorflow.contrib import rnn
# import keras
import numpy as np
import conf

def build_char_cnn_rnn(input_seqs):

    with tf.variable_scope("txt_encode"):
        cnn_dim = 256 # dont know?
        embed_dim = 1024
        alphasize = 70 # like in paper?

        # 201 x alphasize
        conv1 = tf.layers.conv1d(input_seqs, 384, 4, activation=tf.nn.relu, name='txt_conv1')
        max1 = tf.layers.max_pooling1d(conv1, pool_size=3, name='txt_max1', strides=3)

        # 66 x 384
        conv2 = tf.layers.conv1d(max1, 512, 4, activation=tf.nn.relu, name='txt_conv2')
        max2 = tf.layers.max_pooling1d(conv2, pool_size=3, name='txt_max2', strides=3)

        # 21 x 512
        conv3 = tf.layers.conv1d(max2, cnn_dim, 4, activation=tf.nn.relu, name='txt_conv3')
        cnn_out = tf.layers.max_pooling1d(conv3, pool_size=3, name='txt_max3', strides=2)

        # 8 x cnn_dim
        rnn_cell = rnn.BasicRNNCell(cnn_dim, activation=tf.nn.relu, name='txt_rnn_cell') # 1 hidden layer

        # unroll batch
        unstacked =  tf.unstack(cnn_out, axis=1)
        outputs, states = rnn.static_rnn(rnn_cell, tf.unstack(cnn_out, axis=1),  dtype=tf.float32)
        output_stacked = tf.stack(outputs, axis=1)
        normalized = tf.reduce_mean(output_stacked, axis=1)

        out = tf.layers.dense(normalized, embed_dim, name='txt_upscale_dense')

    return out


def downscaler(encoded_text):

    with tf.variable_scope('downscaler'):

        m = 128
        out = tf.layers.dense(encoded_text, m, activation=tf.nn.leaky_relu, name='dense')

        return out


def generator(downscaled_text, z):

    with tf.variable_scope('generator'):

        #nt = 256
        Z = 100 # dimension of noise
        T = 1024 # dimension of text embedding

        # sample noise
        #z = tf.random_normal((Z, 1))

        # Noise concatenated with encoded text
        input = tf.concat([z, downscaled_text], axis = 0)

        conv1 = tf.layers.conv2d_transpose(input, 128*8, (4, 4))
        batch1 = tf.layers.batch_normalization(conv1)

        conv2 = tf.layers.conv2d_transpose(batch1)



        out = 0
        return out


def discriminator(gan_image, encoded_text):

    with tf.variable_scope('discriminator'):

        # something here


        out = 0
        return out






