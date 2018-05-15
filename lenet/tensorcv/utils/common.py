#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Qian Ge <geqian1001@gmail.com>

import math
import os

import tensorflow as tf

__all__ = ['apply_mask', 'apply_mask_inverse', 'get_tensors_by_names',
           'deconv_size', 'match_tensor_save_name']


def apply_mask(input_matrix, mask):
    """Get partition of input_matrix using index 1 in mask.

    Args:
        input_matrix (Tensor): A Tensor
        mask (int): A Tensor of type int32 with indices in {0, 1}. Shape
            has to be the same as input_matrix.

    Return:
        A Tensor with elements from data with entries in mask equal to 1.
    """
    return tf.dynamic_partition(input_matrix, mask, 2)[1]


def apply_mask_inverse(input_matrix, mask):
    """Get partition of input_matrix using index 0 in mask.

    Args:
        input_matrix (Tensor): A Tensor
        mask (int): A Tensor of type int32 with indices in {0, 1}. Shape
            has to be the same as input_matrix.

    Return:
        A Tensor with elements from data with entries in mask equal to 0.
    """
    return tf.dynamic_partition(input_matrix, mask, 2)[0]


def get_tensors_by_names(names):
    """Get a list of tensors by the input name list.

    Args:
        names (str): A str or a list of str

    Return:
        A list of tensors with name in input names.

    Warning:
        If more than one tensor have the same name in the graph. This function
        will only return the tensor with name NAME:0.
    """
    if not isinstance(names, list):
        names = [names]

    graph = tf.get_default_graph()
    tensor_list = []
    # TODO assume there is no repeativie names
    for name in names:
        tensor_name = name + ':0'
        tensor_list += graph.get_tensor_by_name(tensor_name),
    return tensor_list


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
    print('***** WARNING ********: deconv_size is moved to models.utils.py')
    return int(math.ceil(float(input_height) / float(stride))),\
        int(math.ceil(float(input_width) / float(stride)))


def match_tensor_save_name(tensor_names, save_names):
    """
    Match tensor_names and corresponding save_names for saving the results of
    the tenors. If the number of tensors is less or equal to the length
    of save names, tensors will be saved using the corresponding names in
    save_names. Otherwise, tensors will be saved using their own names.
    Used for prediction or inference.

    Args:
        tensor_names (str): List of tensor names
        save_names (str): List of names for saving tensors

    Return:
        (list, list): List of tensor names and list of names to save
        the tensors.
    """
    if not isinstance(tensor_names, list):
        tensor_names = [tensor_names]
    if save_names is None:
        return tensor_names, tensor_names
    elif not isinstance(save_names, list):
        save_names = [save_names]
    if len(save_names) < len(tensor_names):
        return tensor_names, tensor_names
    else:
        return tensor_names, save_names


def check_dir(input_dir):
    print('***** WARNING ********: check_dir is moved to utils.utils.py')
    assert input_dir is not None, "dir cannot be None!"
    assert os.path.isdir(input_dir), input_dir + ' does not exist!'


def assert_type(v, tp):
    print('***** WARNING ********: assert_type is moved to utils.utils.py')
    """
    Assert type of input v be type tp
    """
    assert isinstance(v, tp),\
        "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"
