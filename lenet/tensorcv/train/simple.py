from abc import abstractmethod

import tensorflow as tf

from .config import TrainConfig, GANTrainConfig
from .base import Trainer
from ..callbacks.inputs import FeedInput
from ..callbacks.group import Callbacks
from ..callbacks.hooks import Callback2Hook
from ..models.base import BaseModel, GANBaseModel
from ..utils.sesscreate import ReuseSessionCreator


__all__ = ['SimpleFeedTrainer']

def assert_type(v, tp):
    assert isinstance(v, tp),\
     "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class SimpleFeedTrainer(Trainer):
    """ single optimizer """
    def __init__(self, config):
        assert_type(config.model, BaseModel)
        super(SimpleFeedTrainer, self).__init__(config)

    def _setup(self):
        # TODO to be modified
        cbs = FeedInput(self.dataflow, self.model.get_train_placeholder())

        self.config.callbacks.append(cbs)

        grads = self.model.get_grads()
        opt = self.model.get_optimizer()
        self.train_op = opt.apply_gradients(grads, name='train')

class GANFeedTrainer(Trainer):
    def __init__(self, config):
        assert_type(config, GANTrainConfig)
        # assert_type(config.model, GANBaseModel)

        # config.model.set_batch_size(config.batch_size)

        super(GANFeedTrainer, self).__init__(config)

    def _setup(self):
        # TODO to be modified
        # Since FeedInput only have before_run,
        # it is safe to put this cb only in hooks.
        cbs = FeedInput(self.dataflow, self.model.get_train_placeholder())
        # self.config.callbacks.append(cbs)
        self.feed_input_hook = [Callback2Hook(cbs)]

        dis_grads = self.model.get_discriminator_grads()
        dis_opt = self.model.get_discriminator_optimizer()
        self.dis_train_op = dis_opt.apply_gradients(dis_grads, 
                                        name='discriminator_train')

        gen_grads = self.model.get_generator_grads()
        gen_opt = self.model.get_generator_optimizer()
        self.gen_train_op = gen_opt.apply_gradients(gen_grads, 
                                        name='generator_train')

    def _create_session(self):
        self._dis_callbacks = Callbacks([cb 
                            for cb in self.config.dis_callbacks])
        self._gen_callbacks = Callbacks([cb 
                            for cb in self.config.gen_callbacks])
        dis_hooks = self._dis_callbacks.get_hooks()
        gen_hooks = self._gen_callbacks.get_hooks()

        self.sess = self.config.session_creator.create_session()
        self.dis_hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), 
            hooks=dis_hooks + self.feed_input_hook)
        self.gen_hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), 
            hooks=gen_hooks)

    def _run_step(self):
        model_feed = self.model.get_graph_feed()
        self.dis_hooked_sess.run(self.dis_train_op, feed_dict=model_feed)

        for k in range(0,2):
            model_feed = self.model.get_graph_feed()
            self.gen_hooked_sess.run(self.gen_train_op, feed_dict=ßmodel_feed)












