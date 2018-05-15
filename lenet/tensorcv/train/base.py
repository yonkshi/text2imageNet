from abc import abstractmethod
import weakref
import os

import tensorflow as tf

from .config import TrainConfig
from ..callbacks.base import Callback
from ..callbacks.group import Callbacks
from ..utils.sesscreate import ReuseSessionCreator
from ..callbacks.monitors import TrainingMonitor, Monitors


__all__ = ['Trainer']

def assert_type(v, tp):
    assert isinstance(v, tp),\
     "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Trainer(object):
    """ base class for trainer """
    def __init__(self, config):
        assert_type(config, TrainConfig)
        self._is_load = config.is_load
        self.config = config
        self.model = config.model
        self.model.ex_init_model(config.dataflow, weakref.proxy(self))
        self.dataflow = config.dataflow
        # self.monitors = self.config.monitors
        self._global_step = 0
        self._callbacks = []
        self.monitors = []

        self.default_dirs = config.default_dirs

    @property
    def epochs_completed(self):
        return self.dataflow.epochs_completed

    @property
    def get_global_step(self):
        return self._global_step

    def register_callback(self, cb):
        assert_type(cb, Callback)
        assert not isinstance(self._callbacks, Callbacks), \
        "callbacks have been setup"
        self._callbacks.append(cb)

    def register_monitor(self, monitor):
        assert_type(monitor, TrainingMonitor)
        assert not isinstance(self.monitors, Monitors), \
        "monitors have been setup"
        self.monitors.append(monitor)
        self.register_callback(monitor)


    def _create_session(self):
        hooks = self._callbacks.get_hooks()
        self.sess = self.config.session_creator.create_session()
        
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

        if self._is_load:
            load_model_path = os.path.join(self.config.model_dir, 
                                        self.config.model_name)
            saver = tf.train.Saver()
            saver.restore(self.sess, load_model_path)

    def main_loop(self):
        with self.sess.as_default():
            self._callbacks.before_train()
            while self.epochs_completed <= self.config.max_epoch:
                self._global_step += 1
                print('Epoch: {}. Step: {}'.\
                    format(self.epochs_completed, self._global_step))
                # self._callbacks.before_epoch()
                # TODO to be modified
                self.model.set_is_training(True)
                self._run_step() 
                # self._callbacks.after_epoch()
                self._callbacks.trigger_step()
            self._callbacks.after_train()

    def train(self):
        self.setup()
        self.main_loop()

    @abstractmethod
    def _run_step(self):
        model_feed = self.model.get_graph_feed()
        self.hooked_sess.run(self.train_op, feed_dict=model_feed)

    def setup(self):
        # setup graph from model
        self.setup_graph()
        
        # setup callbacks
        for cb in self.config.callbacks:
            self.register_callback(cb)
        for monitor in self.config.monitors:
            self.register_monitor(monitor)
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))
        self.monitors = Monitors(self.monitors)
        # create session
        self._create_session()

        

        self.sess.graph.finalize()

    def setup_graph(self):
        self.model.create_graph()
        self._setup()
        self.model.setup_summary()
        
    def _setup(self):
        pass






