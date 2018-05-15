import os

import tensorflow as tf

from .base import Callback
from ..utils.common import check_dir

__all__ = ['TrainingMonitor','Monitors','TFSummaryWriter']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class TrainingMonitor(Callback):
    def _setup_graph(self):
        pass

    def process_summary(self, summary):
        self._process_summary(summary)

    def _process_summary(self, summary):
        pass

class Monitors(TrainingMonitor):
    """ group monitors """
    def __init__(self, mons):
        for mon in mons:
            assert_type(mon, TrainingMonitor)
        self.mons = mons

    def _process_summary(self, summary):
        for mon in self.mons:
            mon.process_summary(summary)

class TFSummaryWriter(TrainingMonitor):

    def _setup_graph(self):
        try:
            summary_dir = os.path.join(self.trainer.default_dirs.summary_dir)
            check_dir(summary_dir)
        except AttributeError:
            raise AttributeError('summary_dir is not set in config.py!')
        self._writer = tf.summary.FileWriter(summary_dir)

    def _before_train(self):
        # default to write graph
        self._writer.add_graph(self.trainer.sess.graph)

    def _after_train(self):
        self._writer.close()

    def process_summary(self, summary):
        self._writer.add_summary(summary, self.global_step)
