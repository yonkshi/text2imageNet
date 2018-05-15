import scipy.misc
import os
from abc import ABCMeta
import os

import numpy as np
import tensorflow as tf

from .base import ProxyCallback, Callback

__all__ = ['PeriodicTrigger']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class PeriodicTrigger(ProxyCallback):
	""" may not need """
	def __init__(self, trigger_cb, every_k_steps=None, every_k_epochs=None):

		assert_type(trigger_cb, Callback)
		super(PeriodicTrigger, self).__init__(trigger_cb)
		
		assert (every_k_steps is not None) or (every_k_epochs is not None), \
		"every_k_steps and every_k_epochs cannot be both None!"
		self._k_step = every_k_steps
		self._k_epoch = every_k_epochs

	def __str__(self):
		return 'PeriodicTrigger' + str(self.cb)

	def _trigger_step(self):
		if self._k_step is None:
			return
		if self.global_step % self._k_step == 0:
			self.cb.trigger()

	def _trigger_epoch(self):
		if self._k_epoch is None:
			return
		if self.epochs_completed % self._k_epoch == 0:
			self.cb.trigger()
