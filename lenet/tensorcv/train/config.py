import scipy.misc
import os
import numpy as np

from ..dataflow.base import DataFlow
from ..models.base import ModelDes, GANBaseModel
from ..utils.default import get_default_session_config
from ..utils.sesscreate import NewSessionCreator
from ..callbacks.monitors import TFSummaryWriter
from ..callbacks.summary import TrainSummary
from ..utils.common import check_dir

__all__ = ['TrainConfig', 'GANTrainConfig']

def assert_type(v, tp):
    assert isinstance(v, tp),\
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class TrainConfig(object):
    def __init__(self, 
                 dataflow=None, model=None,
                 callbacks=[],
                 session_creator=None,
                 monitors=None,
                 batch_size=1, max_epoch=100,
                 summary_periodic=None,
                 is_load=False,
                 model_name=None,
                 default_dirs=None):
        self.default_dirs = default_dirs

        assert_type(monitors, TFSummaryWriter), \
        "monitors has to be TFSummaryWriter at this point!"
        if not isinstance(monitors, list):
            monitors = [monitors]
        self.monitors = monitors

        assert dataflow is not None, "dataflow cannot be None!"
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow

        assert model is not None, "model cannot be None!"
        assert_type(model, ModelDes)
        self.model = model
        
        assert batch_size > 0 and max_epoch > 0
        self.dataflow.set_batch_size(batch_size)
        self.model.set_batch_size(batch_size)
        self.batch_size = batch_size
        self.max_epoch = max_epoch 

        self.is_load = is_load
        if is_load:
            assert not model_name is None,\
            '[TrainConfig]: model_name cannot be None when is_load is True!' 
            self.model_name = model_name 
            try:
                self.model_dir = os.path.join(default_dirs.model_dir)
                check_dir(self.model_dir)
            except AttributeError:
                raise AttributeError('model_dir is not set!')   
        
        # if callbacks is None:
        #     callbacks = []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self._callbacks = callbacks

        # TODO model.default_collection only in BaseModel class
        if isinstance(summary_periodic, int):
            self._callbacks.append(
                TrainSummary(key=model.default_collection, 
                            periodic=summary_periodic))

        if session_creator is None:
            self.session_creator = \
               NewSessionCreator(config=get_default_session_config())
        else:
            raise ValueError('custormer session creator is not allowed at this point!')
  
    @property
    def callbacks(self):
        return self._callbacks


class GANTrainConfig(TrainConfig):
    def __init__(self, 
                 dataflow=None, model=None,
                 discriminator_callbacks=[],
                 generator_callbacks=[],
                 session_creator=None,
                 monitors=None,
                 batch_size=1, max_epoch=100,
                 summary_d_periodic=None, 
                 summary_g_periodic=None,
                 default_dirs=None):

        assert_type(model, GANBaseModel)

        if not isinstance(discriminator_callbacks, list):
            discriminator_callbacks = [discriminator_callbacks]
        self._dis_callbacks = discriminator_callbacks
        
        if not isinstance(generator_callbacks, list):
            generator_callbacks = [generator_callbacks]
        self._gen_callbacks = generator_callbacks

        if isinstance(summary_d_periodic, int):
            self._dis_callbacks.append(
                TrainSummary(key=model.d_collection, 
                            periodic=summary_d_periodic))
        if isinstance(summary_g_periodic, int):
            self._dis_callbacks.append(
                TrainSummary(key=model.g_collection, 
                            periodic=summary_g_periodic))

        callbacks = self._dis_callbacks + self._gen_callbacks

        super(GANTrainConfig, self).__init__(
                    dataflow=dataflow, model=model,
                    callbacks=callbacks,
                    session_creator=session_creator,
                    monitors=monitors,
                    batch_size=batch_size, max_epoch=ßmax_epoch,
                    default_dirs=default_dirs)
    @property
    def dis_callbacks(self):
        return self._dis_callbacks
    @property
    def gen_callbacks(self):
        return self._gen_callbacks

