# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

from abc import abstractmethod

import tensorflow as tf
import numpy as np


__all__ = ['BaseBuilder']

class BaseBuilder(object):
    """ base model for model builder """
    def __init__(self):
        self.input = []
        self.output = []
    def Add(self, BaseLayer):
        """ add one layer """
        pass

    # def set_batch_size(self, val):
    #     self._batch_size = val

    # def get_batch_size(self):
    #     return self._batch_size

    # def set_is_training(self, is_training = True):
    #     self.is_training = is_training

    # def get_placeholder(self):
    #     return self._get_placeholder()

    # def _get_placeholder(self):
    #     return []

    # # TODO to be modified
    # def get_prediction_placeholder(self):
    #     return self._get_prediction_placeholder()

    # def _get_prediction_placeholder(self):
    #     return []

    # def get_graph_feed(self):
    #     return self._get_graph_feed()

    # def _get_graph_feed(self):
    #     return {}

    # def create_graph(self):
    #     self._create_graph()
    #     self._setup_graph()
    #     # self._setup_summary()

    # @abstractmethod
    # def _create_graph(self):
    #     raise NotImplementedError()

    # def _setup_graph(self):
    #     pass

    # # TDDO move outside of class
    # # summary will be created before prediction
    # # which is unnecessary
    # def setup_summary(self):
    #     self._setup_summary()

    # def _setup_summary(self):
    #     pass

    
class BaseModel(ModelDes):
    """ Model with single loss and single optimizer """

    def get_optimizer(self):
        try:
            return self.optimizer
        except AttributeError:
            self.optimizer = self._get_optimizer()
        return self.optimizer

    @property
    def default_collection(self):
        return 'default'

    def _get_optimizer(self):
        raise NotImplementedError()

    def get_loss(self):
        try:
            return self._loss
        except AttributeError:
            self._loss = self._get_loss()
            tf.summary.scalar('loss_summary', self.get_loss(), 
                              collections = [self.default_collection])
        return self._loss

    def _get_loss(self):
        raise NotImplementedError()

    def get_grads(self):
        try:
            return self.grads
        except AttributeError:
            optimizer = self.get_optimizer()
            loss = self.get_loss()
            self.grads = optimizer.compute_gradients(loss)
            [tf.summary.histogram('gradient/' + var.name, grad, 
              collections = [self.default_collection]) for grad, var in self.grads]
        return self.grads

class GANBaseModel(ModelDes):
    """ Base model for GANs """
    def __init__(self, input_vec_length, learning_rate):
        self.input_vec_length = input_vec_length
        assert len(learning_rate) == 2
        self.dis_learning_rate, self.gen_learning_rate = learning_rate

    @property
    def g_collection(self):
        return 'default_g'

    @property
    def d_collection(self):
        return 'default_d'

    def get_random_vec_placeholder(self):
        try:
            return self.Z
        except AttributeError:
            self.Z = tf.placeholder(tf.float32, [None, self.input_vec_length])
        return self.Z

    def _get_prediction_placeholder(self):
        return self.get_random_vec_placeholder()

    def get_graph_feed(self):
        default_feed = self._get_graph_feed()
        random_input_feed = self._get_random_input_feed()
        default_feed.update(random_input_feed)
        return default_feed

    def _get_random_input_feed(self):
        feed = {self.get_random_vec_placeholder(): 
                np.random.normal(size = (self.get_batch_size(), 
                                 self.input_vec_length))}
        return feed

    def create_GAN(self, real_data, gen_data_name = 'gen_data'):
        with tf.variable_scope('generator') as scope:
            gen_data = self._generator()
            scope.reuse_variables()
            sample_gen_data = tf.identity(self._generator(train = False), 
                                            name = gen_data_name)
            
        with tf.variable_scope('discriminator') as scope:
            d_real = self._discriminator(real_data)
            scope.reuse_variables()
            d_fake = self._discriminator(gen_data)

        with tf.name_scope('discriminator_out'):
            tf.summary.histogram('discrim_real', 
                                 tf.nn.sigmoid(d_real), 
                                 collections = [self.d_collection])
            tf.summary.histogram('discrim_gen', 
                                  tf.nn.sigmoid(d_fake), 
                                  collections = [self.d_collection])

        return gen_data, sample_gen_data, d_real, d_fake

    # def get_random_input_feed(self):
    #     return self._get_random_input_feed()

    # def _get_random_input_feed(self):
    #     return {}

    def get_discriminator_optimizer(self):
        try:
            return self.d_optimizer
        except AttributeError:
            self.d_optimizer = self._get_discriminator_optimizer()
            return self.d_optimizer

    def get_generator_optimizer(self):
        try:
            return self.g_optimizer
        except AttributeError:
            self.g_optimizer = self._get_generator_optimizer()
            return self.g_optimizer

    def _get_discriminator_optimizer(self):
        raise NotImplementedError()

    def _get_generator_optimizer(self):
        raise NotImplementedError()

    def get_discriminator_loss(self):
        try: 
            return self.d_loss
        except AttributeError:
            self.d_loss = self._get_discriminator_loss()
            tf.summary.scalar('d_loss_summary', self.d_loss, 
                              collections = [self.d_collection])
            return self.d_loss

    def get_generator_loss(self):
        try: 
            return self.g_loss
        except AttributeError:
            self.g_loss = self._get_generator_loss()
            tf.summary.scalar('g_loss_summary', self.g_loss, 
                              collections = [self.g_collection])
            return self.g_loss

    def _get_discriminator_loss(self):
        raise NotImplementedError()

    def _get_generator_loss(self):
        raise NotImplementedError()

    def get_discriminator_grads(self):
        try:
            return self.d_grads
        except AttributeError:
            d_training_vars = [v for v in tf.trainable_variables() 
                               if v.name.startswith('discriminator/')]
            optimizer = self.get_discriminator_optimizer()
            loss = self.get_discriminator_loss()
            self.d_grads = optimizer.compute_gradients(loss, 
                                              var_list = d_training_vars)

            [tf.summary.histogram('d_gradient/' + var.name, grad, 
                        collections = [self.d_collection]) 
                        for grad, var in self.d_grads]
            return self.d_grads

    def get_generator_grads(self):
        try:
            return self.g_grads
        except AttributeError:
            g_training_vars = [v for v in tf.trainable_variables() 
                               if v.name.startswith('generator/')]
            optimizer = self.get_generator_optimizer()
            loss = self.get_generator_loss()
            self.g_grads = optimizer.compute_gradients(loss, 
                                             var_list = g_training_vars)
            [tf.summary.histogram('g_gradient/' + var.name, grad, 
                        collections = [self.g_collection]) 
                        for grad, var in self.g_grads]
            return self.g_grads

    @staticmethod
    def comp_loss_fake(discrim_output):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = discrim_output, 
                                    labels = tf.zeros_like(discrim_output)))

    @staticmethod
    def comp_loss_real(discrim_output):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = discrim_output, 
                                     labels = tf.ones_like(discrim_output)))








