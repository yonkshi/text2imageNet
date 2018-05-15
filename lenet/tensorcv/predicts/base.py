#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import os

import tensorflow as tf

from .config import PridectConfig
from ..utils.sesscreate import ReuseSessionCreator
from ..utils.common import assert_type
from ..callbacks.hooks import Prediction2Hook

__all__ = ['Predictor']


class Predictor(object):
    """Base class for a predictor. Used to run all predictions.

    Attributes:
        config (PridectConfig): the config used for this predictor
        model (ModelDes):
        input (DataFlow):
        sess (tf.Session):
        hooked_sess (tf.train.MonitoredSession):
    """
    def __init__(self, config):
        """ Inits Predictor with config (PridectConfig).

        Will create session as well as monitored sessions for
        each predictions, and load pre-trained parameters.

        Args:
            config (PridectConfig): the config used for this predictor
        """
        assert_type(config, PridectConfig)
        self._config = config
        self._model = config.model

        self._input = config.dataflow
        self._result_dir = config.result_dir

        # TODO to be modified
        self._model.set_is_training(False)
        self._model.create_graph()
        self._restore_vars = self._config.restore_vars

        # pass saving directory to predictions
        for pred in self._config.predictions:
            pred.setup(self._result_dir)

        hooks = [Prediction2Hook(pred) for pred in self._config.predictions]

        self.sess = self._config.session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

        # load pre-trained parameters
        load_model_path = os.path.join(self._config.model_dir,
                                       self._config.model_name)
        if self._restore_vars is not None:
            # variables = tf.contrib.framework.get_variables_to_restore()
            # variables_to_restore = [v for v in variables if v.name.split('/')[0] in self._restore_vars]
            # print(variables_to_restore) 
            saver = tf.train.Saver(self._restore_vars)
        else:
            saver = tf.train.Saver()
        saver.restore(self.sess, load_model_path)

    def run_predict(self):
        """
        Run predictions and the process after finishing predictions.
        """
        with self.sess.as_default():
            self._input.before_read_setup()
            self._predict_step()
            for pred in self._config.predictions:
                pred.after_finish_predict()

            self.after_prediction()

    def _predict_step(self):
        """ Run predictions. Defined in subclass.
        """
        pass

    def after_prediction(self):
        self._after_prediction()

    def _after_prediction(self):
        pass

