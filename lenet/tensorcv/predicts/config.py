import scipy.misc
import os
import numpy as np

from ..dataflow.base import DataFlow
from ..models.base import ModelDes
from ..utils.default import get_default_session_config
from ..utils.sesscreate import NewSessionCreator
from .predictions import PredictionBase
from ..utils.common import check_dir

__all__ = ['PridectConfig']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class PridectConfig(object):
    def __init__(self, 
                 dataflow=None, model=None,
                 model_dir=None, model_name='',
                 restore_vars=None,
                 session_creator=None,
                 predictions=None,
                 batch_size=1,
                 default_dirs=None):
        """
        Args:
        """
        self.model_name = model_name
        try:
            self.model_dir = os.path.join(default_dirs.model_dir)
            check_dir(self.model_dir)
        except AttributeError:
            raise AttributeError('model_dir is not set!')

        try:
            self.result_dir = os.path.join(default_dirs.result_dir)
            check_dir(self.result_dir)
        except AttributeError:
            raise AttributeError('result_dir is not set!')

        if restore_vars is not None:
            if not isinstance(restore_vars, list):
                restore_vars = [restore_vars]
        self.restore_vars = restore_vars


        assert dataflow is not None, "dataflow cannot be None!"
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow
        
        assert batch_size > 0
        self.dataflow.set_batch_size(batch_size)
        self.batch_size = batch_size
        
        assert model is not None, "model cannot be None!"
        assert_type(model, ModelDes)
        self.model = model

        assert predictions is not None, "predictions cannot be None"
        if not isinstance(predictions, list):
            predictions = [predictions]
        for pred in predictions:
            assert_type(pred, PredictionBase)
        self.predictions = predictions
        
        # if not isinstance(callbacks, list):
        #     callbacks = [callbacks]
        # self._callbacks = callbacks

        if session_creator is None:
            self.session_creator = \
                 NewSessionCreator(config=get_default_session_config())
        else:
            raise ValueError('custormer session creator is \
                               not allowed at this point!')
        
    @property
    def callbacks(self):
        return self._callbacks

