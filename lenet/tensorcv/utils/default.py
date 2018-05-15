#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: default.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

__all__ = ['get_default_session_config']


def get_default_session_config(memory_fraction=1):
    """Default config of a TensorFlow session

    Args:
        memory_fraction (float): Memory fraction of GPU for this session

    Return:
        tf.ConfigProto(): Config of session.
    """
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    conf.gpu_options.allow_growth = True

    return conf
