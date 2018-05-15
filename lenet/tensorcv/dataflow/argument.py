#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: argument.py
# Author: Qian Ge <geqian1001@gmail.com>

from .base import DataFlow
from ..utils.utils import assert_type


class ArgumentDataflow(DataFlow):
    def __init__(self, dataflow, argument_order, argument_fnc):
        assert_type(dataflow, DataFlow)
        if not isinstance(argument_order, list):
            argument_order = [argument_order]
        if not isinstance(argument_fnc, list):
            argument_fnc = [argument_fnc]

        self._dataflow = dataflow

        self._order = argument_order
        self._fnc = argument_fnc

    def setup(self, epoch_val, batch_size, **kwargs):
        self._dataflow .setup(epoch_val, batch_size, **kwargs)

    @property
    def epochs_completed(self):
        return self._dataflow.epochs_completed

    def reset_epochs_completed(self, val):
        self._dataflow.reset_epochs_completed(val)

    def set_batch_size(self, batch_size):
        self._dataflow.set_batch_size(batch_size)

    def size(self):
        return self._dataflow.size()

    def reset_state(self):
        self._dataflow.reset_state()

    def after_reading(self):
        self._dataflow.after_reading()

    def next_batch_dict(self):
        batch_data_dict = self._dataflow.next_batch_dict()
        arg_batch_data_dict = {}
        for key, arg_fnc in zip(self._order, self._fnc):
            arg_batch_data_dict[key] = arg_fnc(batch_data_dict[key])

        return arg_batch_data_dict

    def next_batch(self):
        print("***** [ArgumentDataflow.next_batch()] not implmented *****")
