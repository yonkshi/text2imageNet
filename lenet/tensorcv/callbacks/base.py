import scipy.misc
import os
from abc import ABCMeta

import numpy as np
import tensorflow as tf

__all__ = ['Callback', 'ProxyCallback']

def assert_type(v, tp):
    assert isinstance(v, tp),\
     "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Callback(object):
    """ base class for callbacks """

    def setup_graph(self, trainer):
        self.trainer = trainer
        self._setup_graph()

    @property
    def global_step(self):
        return self.trainer.get_global_step

    @property
    def epochs_completed(self):
        return self.trainer.epochs_completed

    def _setup_graph(self):
        pass

    def before_run(self, rct):
        fetch = self._before_run(rct)
        if fetch is None:
            return None
        assert_type(fetch, tf.train.SessionRunArgs)
        return fetch

    def _before_run(self, rct):
        return None

    def after_run(self, rct, val):
        self._after_run(rct, val)

    def _after_run(self, rct, val):
        pass

    def before_train(self):
        self._before_train()

    def _before_train(self):
        pass

    def before_inference(self):
        self._before_inference()

    def _before_inference(self):
        pass

    def after_train(self):
        self._after_train()

    def _after_train(self):
        pass

    def before_epoch(self):
        self._before_epoch()

    def _before_epoch(self):
        pass

    def after_epoch(self):
        self._after_epoch()

    def _after_epoch(self):
        pass

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        self.trigger()

    def trigger_step(self):
        self._trigger_step()

    def _trigger_step(self):
        pass

    def trigger(self):
        self._trigger()

    def _trigger(self):
        pass

    # def before_run(self):

class ProxyCallback(Callback):
    def __init__(self, cb):
        assert_type(cb, Callback)
        self.cb = cb

    def __str__(self):
        return "Proxy-" + str(self.cb)

    def _before_train(self):
        self.cb.before_train()

    def _before_inference(self):
        self.cb.before_inference()

    def _setup_graph(self):
        with tf.name_scope(None):
            self.cb.setup_graph(self.trainer)

    def _trigger_epoch(self):
        self.cb.trigger_epoch()

    def _trigger(self):
        self.cb.trigger()

    def _trigger_step(self):
        self.cb.trigger_step()

    def _after_train(self):
        self.cb.after_train()

    def _before_epoch(self):
        self.cb.before_epoch()

    def _after_epoch(self):
        self.cb.after_epoch()

    def _before_run(self, crt):
        self.cb.before_run(crt)

    def _after_run(self, crt, val):
        self.cb.after_run(crt, val)





    


    
