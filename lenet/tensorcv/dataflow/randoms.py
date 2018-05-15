# File: randoms.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

from .base import DataFlow
from ..utils.utils import get_rng

__all__ = ['RandomVec']

class RandomVec(DataFlow):
    """ random vector input """
    def __init__(self, 
                 len_vec=100):

        self.setup(epoch_val=0, batch_size=1)
        self._len_vec = len_vec

    def next_batch(self):
        self._epochs_completed += 1
        return [np.random.normal(size=(self._batch_size, self._len_vec))]
        
    def size(self):
        return self._batch_size

    def reset_state(self):
        self._reset_state()

    def _reset_state(self):
        self.rng = get_rng(self)

if __name__ == '__main__':
    vec = RandomVec()
    print(vec.next_batch())
    print(vec.next_batch())