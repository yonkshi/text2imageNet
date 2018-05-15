import scipy.misc
import os
import os

import numpy as np
import tensorflow as tf

from .base import Callback
from ..utils.common import check_dir

__all__ = ['ModelSaver']

class ModelSaver(Callback):
    def __init__(self, max_to_keep=5,
                 keep_checkpoint_every_n_hours=0.5,
                 periodic=1,
                 checkpoint_dir=None,
                 var_collections=tf.GraphKeys.GLOBAL_VARIABLES):

        self._periodic = periodic

        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections

    def _setup_graph(self):
        try:
            checkpoint_dir = os.path.join(self.trainer.default_dirs.checkpoint_dir)
            check_dir(checkpoint_dir)
        except AttributeError:
            raise AttributeError('checkpoint_dir is not set in config_path!')

        self._save_path = os.path.join(checkpoint_dir, 'model')
        self._saver = tf.train.Saver()

    def _trigger_step(self):
        if self.global_step % self._periodic == 0:
            self._saver.save(tf.get_default_session(), self._save_path, 
                            global_step = self.global_step)
        


