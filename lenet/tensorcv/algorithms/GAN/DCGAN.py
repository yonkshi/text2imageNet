# File: DCGAN.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse

import tensorflow as tf

from tensorcv.dataflow.randoms import RandomVec
from tensorcv.dataflow.dataset.MNIST import MNIST
# import tensorcv.callbacks as cb
from tensorcv.callbacks import *
from tensorcv.predicts import *
from tensorcv.models.layers import *
from tensorcv.models.losses import *
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.predicts.config import PridectConfig
from tensorcv.models.base import GANBaseModel
from tensorcv.train.config import GANTrainConfig
from tensorcv.train.simple import GANFeedTrainer
from tensorcv.utils.common import deconv_size

import config as config_path


class Model(GANBaseModel):
    def __init__(self,
                 input_vec_length=100,
                 learning_rate=[0.0002, 0.0002],
                 num_channels=None,
                 im_size=None):

        super(Model, self).__init__(input_vec_length, learning_rate)

        if num_channels is not None:
            self.num_channels = num_channels
        if im_size is not None:
            self.im_height, self.im_width = im_size

        self.set_is_training(True)

    # def _get_placeholder(self):
    #     # image
    #     return [self.real_data]

    def _create_input(self):
        self.real_data = tf.placeholder(
            tf.float32,
            [None, self.im_height, self.im_width, self.num_channels])
        self.set_train_placeholder(self.real_data)

    def _generator(self, train=True):

        final_dim = 64
        filter_size = 5

        d_height_2, d_width_2 = deconv_size(self.im_height, self.im_width)
        d_height_4, d_width_4 = deconv_size(d_height_2, d_width_2)
        d_height_8, d_width_8 = deconv_size(d_height_4, d_width_4)
        d_height_16, d_width_16 = deconv_size(d_height_8, d_width_8)

        rand_vec = self.get_random_vec_placeholder()
        batch_size = tf.shape(rand_vec)[0]

        with tf.variable_scope('fc1'):
            fc1 = fc(rand_vec, d_height_16 * d_width_16 * final_dim * 8, 'fc')
            fc1 = tf.nn.relu(batch_norm(fc1, train=train))
            fc1_reshape = tf.reshape(
                fc1, [-1, d_height_16, d_width_16, final_dim * 8])

        with tf.variable_scope('dconv2'):
            output_shape = [batch_size, d_height_8, d_width_8, final_dim * 4]
            dconv2 = dconv(fc1_reshape, filter_size,
                           out_shape=output_shape, name='dconv')
            bn_dconv2 = tf.nn.relu(batch_norm(dconv2, train=train))

        with tf.variable_scope('dconv3'):
            output_shape = [batch_size, d_height_4, d_width_4, final_dim * 2]
            dconv3 = dconv(bn_dconv2, filter_size,
                           out_shape=output_shape, name='dconv')
            bn_dconv3 = tf.nn.relu(batch_norm(dconv3, train=train))

        with tf.variable_scope('dconv4'):
            output_shape = [batch_size, d_height_2, d_width_2, final_dim]
            dconv4 = dconv(bn_dconv3, filter_size,
                           out_shape=output_shape, name='dconv')
            bn_dconv4 = tf.nn.relu(batch_norm(dconv4, train=train))

        with tf.variable_scope('dconv5'):
            # Do not use batch norm for the last layer
            output_shape = [batch_size, self.im_height,
                            self.im_width, self.num_channels]
            dconv5 = dconv(bn_dconv4, filter_size,
                           out_shape=output_shape, name='dconv')

        generation = tf.nn.tanh(dconv5, 'gen_out')
        return generation

    def _discriminator(self, input_im):

        filter_size = 5
        start_depth = 64

        with tf.variable_scope('conv1'):
            conv1 = conv(input_im, filter_size, start_depth, stride=2)
            bn_conv1 = leaky_relu((batch_norm(conv1)))

        with tf.variable_scope('conv2'):
            conv2 = conv(bn_conv1, filter_size, start_depth * 2, stride=2)
            bn_conv2 = leaky_relu((batch_norm(conv2)))

        with tf.variable_scope('conv3'):
            conv3 = conv(bn_conv2, filter_size, start_depth * 4, stride=2)
            bn_conv3 = leaky_relu((batch_norm(conv3)))

        with tf.variable_scope('conv4'):
            conv4 = conv(bn_conv3, filter_size, start_depth * 8, stride=2)
            bn_conv4 = leaky_relu((batch_norm(conv4)))

        with tf.variable_scope('fc5'):
            fc5 = fc(bn_conv4, 1, name='fc')

        return fc5

    def _ex_setup_graph(self):
        tf.identity(self.get_sample_gen_data(), 'generate_image')
        tf.identity(self.get_generator_loss(), 'g_loss_check')
        tf.identity(self.get_discriminator_loss(), 'd_loss_check')

    def _setup_summary(self):
        with tf.name_scope('generator_summary'):
            tf.summary.image('generate_sample',
                             tf.cast(self.get_sample_gen_data(), tf.float32),
                             collections=[self.g_collection])
            tf.summary.image('generate_train',
                             tf.cast(self.get_gen_data(), tf.float32),
                             collections=[self.d_collection])
        with tf.name_scope('real_data'):
            tf.summary.image('real_data',
                             tf.cast(self.real_data, tf.float32),
                             collections=[self.d_collection])


def get_config(FLAGS):
    dataset_train = MNIST('train', data_dir=config_path.data_dir,
                          normalize='tanh')

    inference_list = InferImages('generate_image', prefix='gen')
    random_feed = RandomVec(len_vec=FLAGS.len_vec)

    return GANTrainConfig(
        dataflow=dataset_train,
        model=Model(input_vec_length=FLAGS.len_vec,
                    learning_rate=[0.0002, 0.0002]),
        monitors=TFSummaryWriter(),
        discriminator_callbacks=[
            # ModelSaver(periodic = 100),
            CheckScalar(['d_loss_check', 'g_loss_check'],
                        periodic=10)],
        generator_callbacks=[GANInference(inputs=random_feed,
                                          periodic=100,
                                          inferencers=inference_list)],
        batch_size=FLAGS.batch_size,
        max_epoch=100,
        summary_d_periodic=10,
        summary_g_periodic=10,
        default_dirs=config_path)


def get_predictConfig(FLAGS):
    random_feed = RandomVec(len_vec=FLAGS.len_vec)
    prediction_list = PredictionImage('generate_image',
                                      'test', merge_im=True, tanh=True)
    im_size = [FLAGS.h, FLAGS.w]
    return PridectConfig(dataflow=random_feed,
                         model=Model(input_vec_length=FLAGS.len_vec,
                                     num_channels=FLAGS.input_channel,
                                     im_size=im_size),
                         model_name='model-100',
                         predictions=prediction_list,
                         batch_size=FLAGS.batch_size,
                         default_dirs=config_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--len_vec', default=100, type=int,
                        help='Length of input random vector')
    parser.add_argument('--input_channel', default=1, type=int,
                        help='Number of image channels')
    parser.add_argument('--h', default=28, type=int,
                        help='Heigh of input images')
    parser.add_argument('--w', default=28, type=int,
                        help='Width of input images')
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--predict', action='store_true',
                        help='Run prediction')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')

    return parser.parse_args()


if __name__ == '__main__':

    FLAGS = get_args()

    if FLAGS.train:
        config = get_config(FLAGS)
        GANFeedTrainer(config).train()
    elif FLAGS.predict:
        config = get_predictConfig(FLAGS)
        SimpleFeedPredictor(config).run_predict()
