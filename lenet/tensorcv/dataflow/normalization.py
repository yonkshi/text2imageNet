# File: normalization.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np 


def identity(input_val, *args):
    return input_val

def normalize_tanh(input_val, max_in, half_in):
    return (input_val*1.0 - half_in)/half_in

def normalize_one(input_val, max_in, half_in):
    return input_val*1.0/max_in
