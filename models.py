import tensorflow as tf
from tensorflow.contrib import rnn
# import keras
import numpy as np
import conf

def build_char_cnn_rnn(input_seqs):

    with tf.variable_scope("txt_encode", reuse=tf.AUTO_REUSE):
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

        # unroll batch
        outputs, states = rnn.static_rnn(rnn_cell, tf.unstack(cnn_out, axis=1),  dtype=tf.float32)
        output_stacked = tf.stack(outputs, axis=1)
        normalized = tf.reduce_mean(output_stacked, axis=1)

        out = tf.layers.dense(normalized, embed_dim, name='txt_upscale_dense')

        return out

def generator_resnet(text, enable_res = conf.ENABLE_RESIDUAL_NET, z_size = None):

    with tf.variable_scope('generator_resnet', reuse=tf.AUTO_REUSE):

        #nt = 256
        ngf = conf.NUM_G_FILTER
        m = conf.ENCODED_TEXT_SIZE

        if enable_res:
            w_init = tf.random_normal_initializer(stddev=0.02)
        else:
            w_init = None

        # sample noise
        zz = tf.random_normal((conf.GAN_TOWER_BATCH_SIZE, 100), name='totally_random')
        if z_size is not None: zz = tf.random_normal((z_size, 100))

        # Noise concatenated with encoded text
        downscaled_text = tf.layers.dense(text, m,  activation=tf.nn.relu, name='linear')
        conc = tf.concat([zz, downscaled_text], axis=-1)
        net1 = tf.layers.dense(conc, ngf * 8 * 4 * 4, activation=tf.nn.leaky_relu)
        net1 = tf.reshape(net1, (-1, 4, 4, ngf * 8))
        net1 = tf.layers.batch_normalization(net1)

        # state size: (ngf*8) x 4 x 4
        if enable_res:
            res = tf.layers.conv2d(net1, ngf * 2, kernel_size =(1,1), kernel_initializer=w_init, padding='same')
            res = tf.layers.batch_normalization(res)
            res = tf.nn.relu(res)

            res = tf.layers.conv2d(res, ngf * 2, kernel_size =(3,3), kernel_initializer=w_init,padding='same')
            res = tf.layers.batch_normalization(res)
            res = tf.nn.relu(res)

            res = tf.layers.conv2d(res, ngf * 8, kernel_size=(3,3), kernel_initializer=w_init, padding='same')
            res = tf.layers.batch_normalization(res)

            net1 = tf.nn.relu(res + net1)

        # state size: (ngf*8) x 4 x 4
        net2 = tf.layers.conv2d_transpose(net1, ngf * 4, kernel_size=(4,4), kernel_initializer=w_init, strides = (2,2), padding='same')
        net2 = tf.layers.batch_normalization(net2)

        # state size: (ngf*4) x 8 x 8
        if enable_res:
            res = tf.layers.conv2d(net2, ngf, kernel_size =(1,1), kernel_initializer=w_init, padding='same')
            res = tf.layers.batch_normalization(res)
            res = tf.nn.relu(res)

            res = tf.layers.conv2d(res, ngf, kernel_size =(3,3), kernel_initializer=w_init, padding='same')
            res = tf.layers.batch_normalization(res)
            res = tf.nn.relu(res)

            res = tf.layers.conv2d(res, ngf * 4, kernel_size=(3,3), kernel_initializer=w_init, padding='same')
            res = tf.layers.batch_normalization(res)

            net2 = tf.nn.relu(res + net2)

        # state size: (ngf*4) x 8 x 8

        net3 = tf.layers.conv2d_transpose(net2, ngf * 2, kernel_size=(4, 4), kernel_initializer=w_init, strides = (2,2), padding='same')
        net3 = tf.layers.batch_normalization(net3)
        net3 = tf.nn.relu(net3)

        # state size: (ngf*2) x 16 x 16
        net4 = tf.layers.conv2d_transpose(net3, ngf , kernel_size=(4, 4), kernel_initializer=w_init, strides = (2,2), padding='same')
        net4 = tf.layers.batch_normalization(net4)
        net4 = tf.nn.relu(net4)

        # state size: (ngf) x 32 x 32
        net5 = tf.layers.conv2d_transpose(net4, 3 , kernel_size=(4, 4), kernel_initializer=w_init, strides = (2,2), padding='same')
        net5 = tf.layers.batch_normalization(net5)
        net5 = tf.nn.tanh(net5)

        return net5

def discriminator_resnet(gan_image, text, enable_res = conf.ENABLE_RESIDUAL_NET):
    with tf.variable_scope('discriminator_resnet', reuse=tf.AUTO_REUSE):
        m = 128
        ndf = conf.NUM_D_FILTER
        if enable_res:
            w_init = tf.random_normal_initializer(stddev=0.02)
        else:
            w_init = None
        # Text input
        #txt = tf.layers.dense(text, m,)
        txt = tf.reshape(tf.layers.dense(text, m), [-1, 1, 1, m])
        txt = tf.layers.batch_normalization(txt)
        txt = tf.nn.leaky_relu(txt)
        # nn.Replicate(4,3)
        # nn.Replicate(4,4)
        txt = tf.tile(txt, [1, 4, 4, 1])

        # image imput
        # input is (nc) x 64 x 64
        img = tf.reshape(gan_image, (-1, 64, 64, 3)) # Image size
        img = tf.layers.conv2d(img, ndf, (4,4),
                               strides=(2,2),
                               kernel_initializer=w_init,
                               activation=tf.nn.leaky_relu)

        # state size: (ndf) x 32 x 32
        img = tf.layers.conv2d(img, ndf * 2,(4, 4),
                               strides=(2, 2),
                               kernel_initializer=w_init,
                               padding='same',)
        img = tf.layers.batch_normalization(img)
        img = tf.nn.leaky_relu(img)

        # state size: (ndf*2) x 16 x 16
        img = tf.layers.conv2d(img, ndf * 4, (4, 4),
                                     strides=(2, 2),
                               kernel_initializer=w_init,
                                     padding='same',)
        img = tf.layers.batch_normalization(img)

        # state size: (ndf*4) x 8 x 8
        img = tf.layers.conv2d(img, ndf * 8, (4, 4),
                                     strides=(2, 2),
                               kernel_initializer=w_init,
                                     padding='same',)
        img = tf.layers.batch_normalization(img)

        if enable_res:
            # state size: (ndf*8) x 4 x 4
            res_img = tf.layers.conv2d(img, ndf * 2, (1,1), kernel_initializer=w_init,)
            res_img = tf.layers.batch_normalization(res_img)
            res_img = tf.nn.leaky_relu(res_img)

            res_img = tf.layers.conv2d(res_img, ndf * 2, (3,3), kernel_initializer=w_init,)
            res_img = tf.layers.batch_normalization(res_img)
            res_img = tf.nn.leaky_relu(res_img)

            res_img = tf.layers.conv2d_transpose(res_img, ndf * 8, (3,3), kernel_initializer=w_init,)
            res_img = tf.layers.batch_normalization(res_img)

            img = tf.nn.leaky_relu(res_img + img)

        # descriminator begins
        # state size: (ndf*8 + 128) x 4 x 4
        dnet = tf.concat([img, txt], axis=-1)
        dnet = tf.layers.conv2d(dnet, ndf * 8, (1,1), kernel_initializer=w_init,)
        dnet = tf.layers.batch_normalization(dnet)
        dnet = tf.nn.leaky_relu(dnet)

        dnet = tf.layers.conv2d(dnet, 1, (4,4), kernel_initializer=w_init,)
        dnet = tf.nn.sigmoid(dnet)

    return dnet

def generator(text, z_size=None):

    """
    Generator network
    :param text: encoded input batch    ~ batch_size x 1024
    :param z: sampled noise             ~ batch_size x 100
    :return: synthesized network        ~ batch_size x 64 x 64 x 3
    """

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

        # side length of input to first conv layer
        s = 4

        z = tf.random_normal((conf.GAN_TOWER_BATCH_SIZE, 100), name='totally_random')
        if z_size is not None: z = tf.random_normal((z_size, 100))

        # channel depth in different
        n1 = 1024; n2 = 512; n3 = 256; n4 = 128; channels = 3

        # Dimension of compressed text
        m = 128

        linear = tf.layers.dense(text, m, activation=tf.nn.leaky_relu, name='linear')

        noisy_input = tf.concat([z, linear], axis = -1)


        conv_input = tf.layers.dense(noisy_input, n1*4*4, activation=tf.nn.relu, name='dense_upscale')
        conv_input_reshaped = tf.reshape(conv_input, [-1, 4, 4, n1])

        # 4 x 4 x 1024
        conv1 = tf.layers.conv2d_transpose(conv_input_reshaped, n2, kernel_size=(5,5), strides = (2,2), padding='same', name='conv1')
        batch1 = tf.nn.relu(tf.layers.batch_normalization(conv1, name='batch1'))

        # 8 x 8 x 512
        conv2 = tf.layers.conv2d_transpose(batch1, n3, kernel_size=(5,5), strides=(2,2), padding='same', name='conv2')
        batch2 = tf.nn.relu(tf.layers.batch_normalization(conv2, name='batch2'))

        # 16 x 16 x 256
        conv3 = tf.layers.conv2d_transpose(batch2, n4, kernel_size=(5,5), strides=(2,2), padding='same', name='conv3')
        batch3 = tf.nn.relu(tf.layers.batch_normalization(conv3, name='batch3'))

        # 32 x 32 x 128
        conv4 = tf.layers.conv2d_transpose(batch3, channels, kernel_size=(5,5), strides=(2,2), padding='same', name='conv4')
        batch4 = tf.layers.batch_normalization(conv4, name='batch4')

        # 64 x 64 x 3
        out = tf.nn.tanh(batch4, name='image_output')

        return out


def discriminator(image, text):

    """
    Discriminator network
    :param image: image                 ~ batch_size x 64 x 64 x 3
    :param text: encoded input batch    ~ batch_size x 1024
    :return: probability for True
    """

    # z     : batch_size x 100

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

        # Number of filters
        n1 = 128; n2 = 256; n3 = 512; n4 = 1024

        # kernels and strides
        k = (5,5); s = (2,2)

        # Dimension of compressed text
        m = 128
        img = tf.reshape(image, (-1, 64, 64, 3))
        # 64 x 64 x 3 Going in
        conv1 = tf.layers.conv2d(img, n1, kernel_size=k, strides=s, padding = 'same', name='conv1')
        batch1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, name='batch1'))

        # 32 x 32 x 128 Going in
        conv2 = tf.layers.conv2d(batch1, n2, kernel_size=k, strides=s, padding = 'same', name='conv2')
        batch2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, name='batch2'))

        # 16 x 16 x 256 Going in
        conv3 = tf.layers.conv2d(batch2, n3, kernel_size=k, strides=s, padding = 'same', name='conv3')
        batch3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, name='batch3'))

        # 8 x 8 x 512 Going in
        conv4 = tf.layers.conv2d(batch3, n4, kernel_size=k, strides=s, padding = 'same', name='conv4')
        batch4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, name='batch4'))

        # compress text and the make into matrix. tiled is 4 x 4 x 128
        text = tf.reshape(tf.layers.dense(text, m), [-1, 1, 1, m])
        tiled = tf.tile(text, [1, 4, 4, 1])

        # Concatenate convoluted image and tiled version of text depthwise
        concat = tf.concat([batch4, tiled], axis=-1)

        # 4 x 4 x (1024 + 128) Going in
        conv5 = tf.layers.conv2d(concat, n4, kernel_size=(1,1), name='conv5')
        batch5 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv5, name='batch5'))

        # 4 x 4 x 1024 Going in
        out = tf.nn.sigmoid(tf.layers.conv2d(batch5, 1, kernel_size=batch5.shape[1:3]), name='output')

        # output is probability for True
        return out





