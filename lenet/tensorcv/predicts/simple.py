# File: predictions.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from .base import Predictor 

__all__ = ['SimpleFeedPredictor']

def assert_type(v, tp):
    assert isinstance(v, tp),\
     "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class SimpleFeedPredictor(Predictor):
    """ predictor with feed input """
    # set_is_training
    def __init__(self, config):
        super(SimpleFeedPredictor, self).__init__(config)
        # TODO change len_input to other
        placeholders = self._model.get_prediction_placeholder()
        if not isinstance(placeholders, list):
            placeholders = [placeholders]
        self._plhs = placeholders
        # self.placeholder = self._model.get_random_vec_placeholder()
        # assert self.len_input <= len(self.placeholder)
        # self.placeholder = self.placeholder[0:self.len_input]

    def _predict_step(self):
        while self._input.epochs_completed < 1:
            try:
                cur_batch = self._input.next_batch()
            except AttributeError:
                cur_batch = self._input.next_batch()

            feed = dict(zip(self._plhs, cur_batch))
            self.hooked_sess.run(fetches=[], feed_dict=feed)
        self._input.reset_epochs_completed(0)

    def _after_prediction(self):
        self._input.after_reading()

 


    






