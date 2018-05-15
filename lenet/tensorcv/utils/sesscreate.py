#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: default.py
# Author: Qian Ge <geqian1001@gmail.com>
# Modified from https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/tfutils/sesscreate.py

import tensorflow as tf

from .default import get_default_session_config

__all__ = ['NewSessionCreator', 'ReuseSessionCreator']


class NewSessionCreator(tf.train.SessionCreator):
    """
    tf.train.SessionCreator for a new session
    """
    def __init__(self, target='', graph=None, config=None):
        """ Inits NewSessionCreator with targe, graph and config.

        Args:
            target: same as :meth:`tf.Session.__init__()`.
            graph: same as :meth:`tf.Session.__init__()`.
            config: same as :meth:`tf.Session.__init__()`. Default to
                :func:`utils.default.get_default_session_config()`.
        """
        self.target = target
        if config is not None:
            self.config = config
        else:
            self.config = get_default_session_config()
        self.graph = graph

    def create_session(self):
        """Create session as well as initialize global and local variables

        Return:
            A tf.Session object containing nodes for all of the
            operations in the underlying TensorFlow graph.
        """
        sess = tf.Session(target=self.target,
                          graph=self.graph, config=self.config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess


class ReuseSessionCreator(tf.train.SessionCreator):
    """
    tf.train.SessionCreator for reuse an existed session
    """
    def __init__(self, sess):
        """ Inits ReuseSessionCreator with an existed session.

        Args:
            sess (tf.Session): an existed tf.Session object
        """
        self.sess = sess

    def create_session(self):
        """Create session by reusing an existing session

        Return:
            A reused tf.Session object containing nodes for all of the
            operations in the underlying TensorFlow graph.
        """
        return self.sess
