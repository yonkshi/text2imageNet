#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Qian Ge <geqian1001@gmail.com>

import math


def deconv_size(input_height, input_width, stride=2):
    """
    Compute the feature size (height and width) after filtering with
    a specific stride. Mostly used for setting the shape for deconvolution.

    Args:
        input_height (int): height of input feature
        input_width (int): width of input feature
        stride (int): stride of the filter

    Return:
        (int, int): Height and width of feature after filtering.
    """
    return int(math.ceil(float(input_height) / float(stride))),\
        int(math.ceil(float(input_width) / float(stride)))