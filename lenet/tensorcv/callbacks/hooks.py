import tensorflow as tf

from .base import Callback
from .inferencer import InferencerBase
from ..predicts.predictions import PredictionBase

__all__ = ['Callback2Hook', 'Infer2Hook', 'Prediction2Hook']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Callback2Hook(tf.train.SessionRunHook):
    """ """   
    def __init__(self, cb):
        self.cb = cb

    def before_run(self, rct):
        return self.cb.before_run(rct)

    def after_run(self, rct, val):
        self.cb.after_run(rct, val)

class Infer2Hook(tf.train.SessionRunHook):
	
	def __init__(self, inferencer):
		# to be modified 
		assert_type(inferencer, InferencerBase)
		self.inferencer = inferencer

	def before_run(self, rct):
		return tf.train.SessionRunArgs(fetches=self.inferencer.put_fetch())

	def after_run(self, rct, val):
		self.inferencer.get_fetch(val)

class Prediction2Hook(tf.train.SessionRunHook):
	def __init__(self, prediction):
		assert_type(prediction, PredictionBase)
		self.prediction = prediction

	def before_run(self, rct):
		
		return tf.train.SessionRunArgs(fetches=self.prediction.get_predictions())

	def after_run(self, rct, val):
		self.prediction.after_prediction(val.results)
		


    





    


    
