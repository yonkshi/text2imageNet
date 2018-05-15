from abc import abstractmethod

import tensorflow as tf
import numpy as np

from .losses import *


__all__ = ['ModelDes', 'BaseModel', 'GANBaseModel']

class ModelDes(object):
    """ base model for ModelDes """

    def ex_init_model(self, dataflow, trainer):
        self.trainer = trainer
        # may move in create_graph
        try:
            self.im_height = dataflow.im_size[0]
            self.im_width = dataflow.im_size[1]
        except AttributeError:
            pass
            # self.im_height = self.im_height
            # self.im_width = self.im_height
        try:
            self.num_channels = dataflow.num_channels
        except AttributeError:
            pass
            # self.num_channels = self.num_channels

    @property
    def get_global_step(self):
        return self.trainer.get_global_step


    def set_batch_size(self, val):
        self._batch_size = val

    def get_batch_size(self):
        return self._batch_size

    def set_is_training(self, is_training=True):
        self.is_training = is_training

    def get_train_placeholder(self):
        default_plh = self._get_train_placeholder()
        if not isinstance(default_plh, list):
            default_plh = [default_plh]
        try:
            return self._train_plhs + default_plh
        except AttributeError:
            return default_plh

    def _get_train_placeholder(self):
        return []

    def set_train_placeholder(self, plhs=None):
        if not isinstance(plhs, list):
            plhs = [plhs]
        self._train_plhs = plhs

    # TODO to be modified
    def get_prediction_placeholder(self):
        default_plh = self._get_prediction_placeholder()
        if not isinstance(default_plh, list):
            default_plh = [default_plh]
        try:
            return self._predict_plhs + default_plh
        except AttributeError:
            return default_plh

    def _get_prediction_placeholder(self):
        return []

    def set_prediction_placeholder(self, plhs=None):
        if not isinstance(plhs, list):
            plhs = [plhs]
        self._predict_plhs = plhs

    def get_graph_feed(self):
        return self._get_graph_feed()

    def _get_graph_feed(self):
        """ return keep_prob feed when dropout is set """
        try:
            if self.is_training:
                feed = {self._dropout_pl: self._keep_prob}
            else:
                feed = {self._dropout_pl: 1}
            return feed
        except AttributeError:
            return {}

    def set_dropout(self, dropout_placeholder, keep_prob=0.5):
        self._dropout_pl = dropout_placeholder
        self._keep_prob = keep_prob

    def create_graph(self):
        # self._create_graph()
        # self._setup_graph()

        self._create_input()
        self._create_model()
        self._ex_setup_graph()

    def create_model(self, inputs=None):
        print('**[warning]** consider use dictionary input.')
        """ only called when defined inside other model"""
        assert inputs is not None, 'inputs cannot be None!'
        if not isinstance(inputs, list):
            inputs = [inputs]
        self._input = inputs
        self._create_model()


    @abstractmethod
    def _create_model(self):
        raise NotImplementedError()

    @abstractmethod
    def _create_input(self):
        raise NotImplementedError() 

    @property
    def model_input(self):
        try:
            return self._input
        except AttributeError:
            raise AttributeError

    def set_model_input(self, inputs=None):
        self._input = inputs

    

    # def _get_model_input(self):
    #     return []

    @abstractmethod
    def _create_graph(self):
        raise NotImplementedError()

    def _ex_setup_graph(self):
        pass

    # def _setup_graph(self):
    #     pass

    # TDDO move outside of class
    # summary will be created before prediction
    # which is unnecessary
    def setup_summary(self):
        self._setup_summary()

    def _setup_summary(self):
        pass

    
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

    def _create_model(self):
        # TODO
        real_data = self.get_train_placeholder()[0]

        with tf.variable_scope('generator') as scope:
            self.gen_data = self._generator()
            scope.reuse_variables()
            self.sample_gen_data = self._generator(train = False)
            
        with tf.variable_scope('discriminator') as scope:
            self.d_real = self._discriminator(real_data)
            scope.reuse_variables()
            self.d_fake = self._discriminator(self.gen_data)

        with tf.name_scope('discriminator_out'):
            tf.summary.histogram('discrim_real', 
                                 tf.nn.sigmoid(self.d_real), 
                                 collections = [self.d_collection])
            tf.summary.histogram('discrim_gen', 
                                  tf.nn.sigmoid(self.d_fake), 
                                  collections = [self.d_collection])
    def get_gen_data(self):
        return self.gen_data

    def get_sample_gen_data(self):
        return self.sample_gen_data

    def def_loss(self, dis_loss_fnc, gen_loss_fnc):
        """ updata definintion of loss functions """
        self.d_loss = dis_loss_fnc(self.d_real, self.d_fake, name='d_loss')
        self.g_loss = gen_loss_fnc(self.d_fake, name='g_loss')


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
        # TODO use for future
        self.d_optimizer = tf.train.AdamOptimizer(beta1=0.5,
                        learning_rate=self.dis_learning_rate)
        return self.d_optimizer

    def _get_generator_optimizer(self):
        # TODO use for future
        self.g_optimizer = tf.train.AdamOptimizer(beta1=0.5,
                        learning_rate=self.gen_learning_rate)
        return self.g_optimizer

    def get_discriminator_loss(self):
        try: 
            return self.d_loss
        except AttributeError:
            self.d_loss = self._get_discriminator_loss()
            tf.summary.scalar('d_loss_summary', self.d_loss, 
                              collections=[self.d_collection])
            return self.d_loss

    def get_generator_loss(self):
        try: 
            return self.g_loss
        except AttributeError:
            self.g_loss = self._get_generator_loss()
            tf.summary.scalar('g_loss_summary', self.g_loss, 
                              collections=[self.g_collection])
            return self.g_loss

    def _get_discriminator_loss(self):
        return GAN_discriminator_loss(self.d_real, self.d_fake, 
                                    name='d_loss')

    def _get_generator_loss(self):
        return GAN_generator_loss(self.d_fake, name='g_loss')

    def get_discriminator_grads(self):
        try:
            return self.d_grads
        except AttributeError:
            d_training_vars = [v for v in tf.trainable_variables() 
                               if v.name.startswith('discriminator/')]
            optimizer = self.get_discriminator_optimizer()
            loss = self.get_discriminator_loss()
            self.d_grads = optimizer.compute_gradients(loss, 
                                              var_list=d_training_vars)

            [tf.summary.histogram('d_gradient/' + var.name, grad, 
                        collections=[self.d_collection]) 
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
                                             var_list=g_training_vars)
            [tf.summary.histogram('g_gradient/' + var.name, grad, 
                        collections=[self.g_collection]) 
                        for grad, var in self.g_grads]
            return self.g_grads









