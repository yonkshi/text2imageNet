#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import matplotlib.pyplot as plt

from .sequence import SeqDataflow
from ..utils.utils import assert_type

def plot_seq(dataflow, data_range=None):
    assert_type(dataflow, SeqDataflow)
    data = dataflow.get_entire_seq()
    if data_range is None:
        plt.plot(data)
    else:
        plt.plot(data[data_range])
    plt.show()
