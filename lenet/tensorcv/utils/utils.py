#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
from datetime import datetime
import numpy as np

__all__ = ['get_rng']

_RNG_SEED = None


def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


def check_dir(input_dir):
    assert input_dir is not None, "dir cannot be None!"
    assert os.path.isdir(input_dir), input_dir + ' does not exist!'


def assert_type(v, tp):
    """
    Assert type of input v be type tp
    """
    assert isinstance(v, tp),\
        "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"
