from abc import abstractmethod, ABCMeta
import numpy as np 

from ..utils.utils import get_rng

__all__ = ['DataFlow', 'RNGDataFlow']

# @six.add_metaclass(ABCMeta)
class DataFlow(object):
    """ base class for dataflow """
    # self._epochs_completed = 0

    def before_read_setup(self, **kwargs):
        pass

    def setup(self, epoch_val, batch_size, **kwargs):
        self.reset_epochs_completed(epoch_val)
        self.set_batch_size(batch_size)
        self.reset_state()
        self._setup()

    def _setup(self, **kwargs):
        pass

    # @property
    # def channels(self):
    #     try:
    #         return self._num_channels
    #     except AttributeError:
    #         self._num_channels = self._get_channels()
    #         return self._num_channels

    # def _get_channels(self):
    #     return 0

    # @property
    # def im_size(self):
    #     try:
    #         return self._im_size
    #     except AttributeError:
    #         self._im_size = self._get_im_size()
    #         return self._im_size

    def _get_im_size(self):
        return 0

    @property
    def epochs_completed(self):
        return self._epochs_completed 

    def reset_epochs_completed(self, val):
        self._epochs_completed  = val

    @abstractmethod
    def next_batch(self):
        return

    def next_batch_dict(self):
        print('Need to be implemented!')

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def size(self):
        raise NotImplementedError()

    def reset_state(self):
        self._reset_state()

    def _reset_state(self):
        pass

    def after_reading(self):
        pass

class RNGDataFlow(DataFlow):
    def _reset_state(self):
        self.rng = get_rng(self)

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        self.file_list = self.file_list[idxs]

    def suffle_data(self):
        self._suffle_file_list()
