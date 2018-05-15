# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

import os

import numpy as np
import tensorflow as tf

from .base import Callback
from ..utils.common import get_tensors_by_names, check_dir, match_tensor_save_name
from ..utils.viz import *

__all__ = ['InferencerBase', 'InferImages', 'InferScalars', 'InferOverlay', 'InferMat']

class InferencerBase(Callback):

    def setup_inferencer(self):
        if not isinstance(self._names, list):
            self._names = [self._names]
        self._names = get_tensors_by_names(self._names)
        self._setup_inference(self.trainer.default_dirs)

    def _setup_inference(self, default_dirs=None):
        pass

    def put_fetch(self):
        return self._put_fetch()

    def _put_fetch(self):
        return self._names
        # pass

    def get_fetch(self, val):
        self._get_fetch(val)

    def _get_fetch(self, val):
        pass

    def before_inference(self):
        """ process before every inference """
        self._before_inference()

    def _before_inference(self):
        pass

    def after_inference(self):
        self._after_inference()

        # if re is not None:
        #     for key, val in re.items():
        #         s = tf.Summary()
        #         s.value.add(tag = key, simple_value = val)
        #         self.trainer.monitors.process_summary(s)

    def _after_inference(self):
        return None



class InferImages(InferencerBase):
    def __init__(self, im_name, prefix=None, color=False, tanh=False):
        self._names, self._prefix = match_tensor_save_name(im_name, prefix)
        self._color = color
        self._tanh = tanh

    def _setup_inference(self, default_dirs=None):
        try:
            self._save_dir = os.path.join(self.trainer.default_dirs.infer_dir)
            check_dir(self._save_dir)
        except AttributeError:
            raise AttributeError('summary_dir is not set in infer_dir.py!')

    def _before_inference(self):
        self._result_list = []
        
    def _get_fetch(self, val):
        self._result_list.append(val.results)
        # self._result_im = val.results

    def _after_inference(self):
        # TODO add process_image to monitors
        # batch_size = len(self._result_im[0])
        batch_size = len(self._result_list[0][0])
        grid_size = self._get_grid_size(batch_size)
        # grid_size = [8, 8] if batch_size == 64 else [6, 6]
        local_step = 0
        for result_im in self._result_list:
            for im, save_name in zip(result_im, self._prefix): 
                save_merge_images(im, [grid_size, grid_size], 
                    self._save_dir + save_name + '_step_' + str(self.global_step) +\
                    '_b_' + str(local_step) + '.png',
                    color = self._color,
                    tanh = self._tanh)
            local_step += 1
        return None

    def _get_grid_size(self, batch_size):
        try:
            return self._grid_size 
        except AttributeError:
            self._grid_size = np.ceil(batch_size**0.5).astype(int)
            return self._grid_size

class InferOverlay(InferImages):
    def __init__(self, im_name, prefix=None, color=False, tanh=False):
        if not isinstance(im_name, list):
            im_name = [im_name]
        assert len(im_name) == 2,\
        '[InferOverlay] requires two image tensors but the input len = {}.'.\
        format(len(im_name))
        super(InferOverlay, self).__init__(im_name=im_name,
                                           prefix=prefix,
                                           color=color,
                                           tanh=tanh)
        self._overlay_prefix = '{}_{}'.format(self._prefix[0], self._prefix[1])

    def _after_inference(self):
        # TODO add process_image to monitors
        # batch_size = len(self._result_im[0])
        batch_size = len(self._result_list[0][0])
        grid_size = self._get_grid_size(batch_size)
        # grid_size = [8, 8] if batch_size == 64 else [6, 6]
        local_step = 0
        for result_im in self._result_list:
            overlay_im_list = []
            for im_1, im_2 in zip(result_im[0], result_im[1]):
                overlay_im = image_overlay(im_1, im_2, color = self._color)
                overlay_im_list.append(overlay_im)
                save_merge_images(np.squeeze(overlay_im_list), [grid_size, grid_size], 
                self._save_dir + self._overlay_prefix + '_step_' +str(self.global_step) +\
                '_b_' + str(local_step) + '.png',
                color = False, tanh = self._tanh)
            local_step += 1
        return None

class InferMat(InferImages):
    def __init__(self, infer_save_name, mat_name, prefix=None):
        self._infer_save_name = str(infer_save_name)
        super(InferMat, self).__init__(im_name = mat_name, prefix=prefix, 
                                        color=False, tanh=False)
    def _after_inference(self):
        for idx, batch_result in enumerate(self._result_list):
            save_path = os.path.join(self._save_dir, 
                '{}_step_{}_b_{}.mat'.format(self._infer_save_name, self.global_step, idx))
                               # self._infer_save_name + '_b_' + str(idx) + str(self.global_step) + '.mat')
            scipy.io.savemat(save_path, {name: np.squeeze(val) for name, val 
              in zip(self._prefix, batch_result)})
        return None

class InferScalars(InferencerBase):
    def __init__(self, scaler_names, summary_names=None):
        if not isinstance(scaler_names, list): 
            scaler_names = [scaler_names]
        self._names = scaler_names
        if summary_names is None:
            self._summary_names = scaler_names
        else:
            if not isinstance(summary_names, list): 
                summary_names = [summary_names]
            assert len(self._names) == len(summary_names), \
            "length of scaler_names and summary_names has to be the same!"
            self._summary_names = summary_names 
        
    def _before_inference(self):
        self.result_list = [[] for i in range(0, len(self._names))]

    def _get_fetch(self, val):
        for i,v in enumerate(val.results):
            self.result_list[i] += v,

    def _after_inference(self):
        """ process after get_fetch """
        summary_dict = {name: np.mean(val) for name, val 
                        in zip(self._summary_names, self.result_list)}
        if summary_dict is not None:
            for key, val in summary_dict.items():
                s = tf.Summary()
                s.value.add(tag=key, simple_value=val)
                self.trainer.monitors.process_summary(s)
                print('[infer] '+ key + ': ' + str(val))
        # return {name: np.mean(val) for name, val 
        #       in zip(self._summary_names, self.result_list)}

# TODO to be modified
# class BinaryClassificationStats(InferencerBase):
#     def __init__(self, accuracy):
#         self._names = accuracy
        
#     def _before_inference(self):
#         self.result_list = []

#     def _put_fetch(self):
#         # fetch_list = self.names
#         return self._names

#     def _get_fetch(self, val):
#         self.result_list += val.results,

#     def _after_inference(self):
#         """ process after get_fetch """
#         return {"test_accuracy": np.mean(self.result_list)}

if __name__ == '__main__':
    t = InferGANGenerator('gen_name', 
            save_dir = 'D:\\Qian\\GitHub\\workspace\\test\\result\\', prefix = 1)
    print(t._prefix)

    

