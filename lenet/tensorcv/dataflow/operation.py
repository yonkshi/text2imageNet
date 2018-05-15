#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: operation.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import copy

from .base import DataFlow
from ..utils.utils import assert_type


def display_dataflow(dataflow, data_name='data', simple=False):
    assert_type(dataflow, DataFlow)

    n_sample = dataflow.size()
    try:
        label_list = dataflow.get_label_list()
        n_class = len(set(label_list))
        print('[{}] num of samples {}, num of classes {}'.
              format(data_name, n_sample, n_class))
        if not simple:
            nelem_dict = {}
            for c_label in label_list:
                try:
                    nelem_dict[c_label] += 1
                except KeyError:
                    nelem_dict[c_label] = 1
            for c_label in nelem_dict:
                print('class {}: {}'.format(
                    dataflow.label_dict_reverse[c_label],
                    nelem_dict[c_label]))
    except AttributeError:
        print('[{}] num of samples {}'.
              format(data_name, n_sample))


def k_fold_based_class(dataflow, k, shuffle=True):
    """Partition dataflow into k equal sized subsamples based on class labels

    Args:
        dataflows (DataFlow): DataFlow to be partitioned. Must contain labels.
        k (int): number of subsamples
        shuffle (bool): data will be shuffled before and after partition
            if is true

    Return:
        DataFlow: list of k subsample Dataflow
    """
    assert_type(dataflow, DataFlow)
    k = int(k)
    assert k > 0, 'k must be an integer grater than 0!'
    dataflow_data_list = dataflow.get_data_list()
    if not isinstance(dataflow_data_list, list):
        dataflow_data_list = [dataflow_data_list]

    label_list = dataflow.get_label_list()
    # im_list = dataflow.get_data_list()

    if shuffle:
        dataflow.suffle_data()

    class_dict = {}
    for idx, cur_label in enumerate(label_list):
        try:
            for data_idx, data in enumerate(dataflow_data_list):
                class_dict[cur_label][data_idx] += [data[idx], ]
        except KeyError:
            class_dict[cur_label] = [[] for i in range(0, len(dataflow_data_list))]
            for data_idx, data in enumerate(dataflow_data_list):
                class_dict[cur_label][data_idx] = [data[idx], ]
            # class_dict[cur_label] = [cur_im, ]

    # fold_im_list = [[] for i in range(0, k)]
    fold_data_list = [[[] for j in range(0, len(dataflow_data_list))] for i in range(0, k)]
    # fold_label_list = [[] for i in range(0, k)]
    for label_key in class_dict:
        cur_data_list = class_dict[label_key]
        nelem = int(len(cur_data_list[0]) / k)
        start_id = 0
        for fold_id in range(0, k-1):
            for data_idx, data_list in enumerate(cur_data_list):
                fold_data_list[fold_id][data_idx].extend(data_list[start_id : start_id + nelem])
            start_id += nelem
        for data_idx, data_list in enumerate(cur_data_list):
            fold_data_list[k - 1][data_idx].extend(data_list[start_id :])

    data_folds = [copy.deepcopy(dataflow) for i in range(0, k)]

    for cur_fold, cur_data_list in zip(data_folds, fold_data_list):
        cur_fold.set_data_list(cur_data_list)

        if shuffle:
            cur_fold.suffle_data()

    return data_folds


def combine_dataflow(dataflows, shuffle=True):
    """Combine several dataflow into one

    Args:
        dataflows (DataFlow list): list of DataFlow to be combined
        shuffle (bool): data will be shuffled after combined if is true

    Return:
        DataFlow: Combined DataFlow
    """
    if not isinstance(dataflows, list):
        dataflows = [dataflows]

    data_list = []
    for cur_dataflow in dataflows:
        assert_type(cur_dataflow, DataFlow)
        cur_data_list = cur_dataflow.get_data_list()
        if not isinstance(cur_data_list, list):
            cur_data_list = [cur_data_list]
        data_list.append(cur_data_list)

    num_data_type = len(data_list[0])
    combined_data_list = [[] for i in range(0, num_data_type)]
    for cur_data_list in data_list:
        for i in range(0, num_data_type):
            combined_data_list[i].extend(cur_data_list[i])

    dataflows[0].set_data_list(combined_data_list)
    if shuffle:
        dataflows[0].suffle_data()

    return dataflows[0]
