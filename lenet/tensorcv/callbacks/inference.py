# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc
import os
from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback
from .group import Callbacks
from .inputs import FeedInput
from ..dataflow.base import DataFlow
from ..dataflow.randoms import RandomVec
from .hooks import Callback2Hook, Infer2Hook
from ..utils.sesscreate import ReuseSessionCreator
from .inferencer import InferencerBase

__all__ = ['FeedInference', 'GANInference', 'FeedInferenceBatch']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class InferenceBase(Callback):
    """ base class for Inference """
    def __init__(self, inputs=None, periodic=1, 
                 inferencers=None, extra_cbs=None,
                 infer_batch_size=None):
        """
        Args:
            extra_cbs (list[Callback])
        """
        self._inputs = inputs
        self._periodic = periodic
        self._infer_batch_size = infer_batch_size

        assert inferencers is not None or extra_cbs is not None,\
        "Inferencers and extra_cbs cannot be both None!"

        if not isinstance(inferencers, list):
            inferencers = [inferencers]
        for infer in inferencers:
            assert_type(infer, InferencerBase)
        self._inference_list = inferencers

        if extra_cbs is None:
            self._extra_cbs = []
        elif not isinstance(extra_cbs, list):
            self._extra_cbs = [extra_cbs]
        else:
            self._extra_cbs = extra_cbs

        self._cbs = []

    def _setup_graph(self):
        self.model = self.trainer.model
        self.setup_inference()
        self.register_cbs()
        self._cbs = Callbacks(self._cbs)
        self._cbs.setup_graph(self.trainer)

        for infer in self._inference_list:
            infer.setup_inferencer()
   
    def setup_inference(self):
        self._setup_inference()

        for infer in self._inference_list:
            assert_type(infer, InferencerBase)
            infer.setup_graph(self.trainer)

        if self._infer_batch_size is None:
            self._inputs.set_batch_size(self.trainer.config.batch_size)
        else:
            self._inputs.set_batch_size(self._infer_batch_size)

    def _setup_inference(self):
        """ setup extra default callbacks for inference """
        pass

    def register_cbs(self):
        for cb in self._extra_cbs:
            assert_type(cb, Callback)
            self._cbs.append(cb)
        
    def get_infer_hooks(self):
        return (self._cbs.get_hooks() 
                + [Infer2Hook(infer) for infer in self._inference_list])
        
    def _create_infer_sess(self):
        self.sess = self.trainer.sess
        infer_hooks = self.get_infer_hooks()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator = ReuseSessionCreator(self.sess), 
            hooks = infer_hooks)

    def _trigger_step(self):
        if self.global_step % self._periodic == 0:
            for infer in self._inference_list:
                infer.before_inference()

            self._create_infer_sess()
            self.inference_step()

            for infer in self._inference_list:
                infer.after_inference()

    def inference_step(self):
        # TODO to be modified
        self.model.set_is_training(False)
        self._cbs.before_inference()
        self._inference_step()

    def _inference_step(self, extra_feed):
        self.hooked_sess.run(fetches = [], feed_dict = extra_feed)

    def _after_train(self):
        self._cbs.after_train()

        
class FeedInference(InferenceBase):
    """
    default inferencer:
        inference_list = InferImages('generator/gen_image', prefix = 'gen')
    """
    def __init__(self, inputs, periodic=1, 
                 inferencers=[], extra_cbs=None,
                 infer_batch_size=None):
        assert_type(inputs, DataFlow)

        # inferencers.append(InferImages('default', prefix = 'gen'))
        super(FeedInference, self).__init__(inputs=inputs, 
                                            periodic=periodic, 
                                            inferencers=inferencers,
                                            extra_cbs=extra_cbs,
                                            infer_batch_size=infer_batch_size)

    def _setup_inference(self):
        placeholders = self.model.get_train_placeholder()
        self._extra_cbs.append(FeedInput(self._inputs, placeholders))

    def _inference_step(self):
        model_feed = self.model.get_graph_feed()
        while self._inputs.epochs_completed <= 0:
            self.hooked_sess.run(fetches = [], feed_dict = model_feed)
        self._inputs.reset_epochs_completed(0)

class FeedInferenceBatch(FeedInference):
    """ do not use all validation data """
    def __init__(self, inputs, periodic=1, 
                 batch_count=10,
                 inferencers=[], extra_cbs=None,
                 infer_batch_size=None):
        self._batch_count = batch_count
        super(FeedInferenceBatch, self).__init__(inputs=inputs, 
                                                periodic=periodic, 
                                                inferencers=inferencers, 
                                                extra_cbs=extra_cbs,
                                                infer_batch_size=infer_batch_size)
    def _inference_step(self):
        model_feed = self.model.get_graph_feed()
        for i in range(self._batch_count):
            self.hooked_sess.run(fetches=[], feed_dict=model_feed)


class GANInference(InferenceBase):
    def __init__(self, inputs=None, periodic=1, 
                 inferencers=None, extra_cbs=None):
        if inputs is not None:
            assert_type(inputs, RandomVec)
        super(GANInference, self).__init__(inputs=inputs, 
                                           periodic=periodic, 
                                           inferencers=inferencers, 
                                           extra_cbs=extra_cbs)

    def _setup_inference(self):
        if self._inputs is not None:
            self._inputs.set_batch_size(self.trainer.config.batch_size)
            rand_vec_phs = self.model.get_random_vec_placeholder()
            self._extra_cbs.append(FeedInput(self._inputs, rand_vec_phs))

    def _inference_step(self):
        if self._inputs is None:
            model_feed = self.model.get_graph_feed()
        else:
            model_feed = {}
        # while self._inputs.epochs_completed <= 0:
        self.hooked_sess.run(fetches=[], feed_dict=model_feed)
        # self._inputs.reset_epochs_completed(0)

