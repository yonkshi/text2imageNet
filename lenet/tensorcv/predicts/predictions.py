# File: predictions.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.io
import scipy.misc

import tensorflow as tf
import numpy as np

from ..utils.common import get_tensors_by_names
from ..utils.viz import *

__all__ = ['PredictionImage', 'PredictionScalar', 'PredictionMat', 'PredictionMeanScalar', 'PredictionOverlay']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class PredictionBase(object):
    """ base class for prediction 

    Attributes:
        _predictions
        _prefix_list
        _global_ind
        _save_dir
    """
    def __init__(self, prediction_tensors, save_prefix):
        """ init prediction object

        Get tensors to be predicted and the prefix for saving 
        each tensors

        Args:
            prediction_tensors : list[string] A tensor name or list of tensor names
            save_prefix: list[string] A string or list of strings
            Length of prediction_tensors and save_prefix have 
            to be the same
        """
        if not isinstance(prediction_tensors, list):
            prediction_tensors = [prediction_tensors]
        if not isinstance(save_prefix, list):
            save_prefix = [save_prefix]
        assert len(prediction_tensors) == len(save_prefix), \
        'Length of prediction_tensors {} and save_prefix {} has to be the same'.\
        format(len(prediction_tensors), len(save_prefix))

        self._predictions = prediction_tensors
        self._prefix_list = save_prefix
        self._global_ind = 0

    def setup(self, result_dir):
        assert os.path.isdir(result_dir)
        self._save_dir = result_dir

        self._predictions = get_tensors_by_names(self._predictions)

    def get_predictions(self):
        return self._predictions

    def after_prediction(self, results):
        """ process after predition
            default to save predictions
        """
        self._save_prediction(results)

    def _save_prediction(self, results):
        pass

    def after_finish_predict(self):
        """ process after all prediction steps """
        self._after_finish_predict()

    def _after_finish_predict(self):
        pass

class PredictionImage(PredictionBase):
    """ Predict image output and save as files.

    Images are saved every batch. Each batch result can be 
    save in one image or individule images.
    
    """
    def __init__(self, prediction_image_tensors, 
                save_prefix, merge_im=False, 
                tanh=False, color=False):
        """
        Args:
            prediction_image_tensors (list): a list of tensor names
            save_prefix (list): a list of file prefix for saving 
                                each tensor in prediction_image_tensors
            merge_im (bool): merge output of one batch or not
        """
        self._merge = merge_im
        self._tanh = tanh
        self._color = color
        super(PredictionImage, self).__init__(prediction_tensors=prediction_image_tensors, 
                                             save_prefix=save_prefix)

    def _save_prediction(self, results):

        for re, prefix in zip(results, self._prefix_list):
            cur_global_ind = self._global_ind
            if self._merge and re.shape[0] > 1:
                grid_size = self._get_grid_size(re.shape[0])
                save_path = os.path.join(self._save_dir, 
                               str(cur_global_ind) + '_' + prefix + '.png')
                save_merge_images(np.squeeze(re), 
                                [grid_size, grid_size], save_path, 
                                tanh=self._tanh, color=self._color)
                cur_global_ind += 1
            else:
                for im in re:
                    save_path = os.path.join(self._save_dir, 
                               str(cur_global_ind) + '_' + prefix + '.png')
                    if self._color:
                        im = intensity_to_rgb(np.squeeze(im), normalize=True)
                    scipy.misc.imsave(save_path, np.squeeze(im))
                    cur_global_ind += 1
        self._global_ind = cur_global_ind

    def _get_grid_size(self, batch_size):
        try:
            return self._grid_size 
        except AttributeError:
            self._grid_size = np.ceil(batch_size**0.5).astype(int)
            return self._grid_size

class PredictionOverlay(PredictionImage):
    def __init__(self, prediction_image_tensors, 
                save_prefix, merge_im=False, 
                tanh=False, color=False):
        if not isinstance(prediction_image_tensors, list):
            prediction_image_tensors = [prediction_image_tensors]
        assert len(prediction_image_tensors) == 2,\
        '[PredictionOverlay] requires two image tensors but the input len = {}.'.\
        format(len(prediction_image_tensors))

        super(PredictionOverlay, self).__init__(prediction_image_tensors, 
                                            save_prefix, merge_im=merge_im, 
                                            tanh=tanh, color=color)

        self._overlay_prefix = '{}_{}'.format(self._prefix_list[0], self._prefix_list[1])

    def _save_prediction(self, results):
        cur_global_ind = self._global_ind

        if self._merge and results[0].shape[0] > 1:
            overlay_im_list = []
            for im_1, im_2 in zip(results[0], results[1]):
                overlay_im = image_overlay(im_1, im_2, color=self._color)
                overlay_im_list.append(overlay_im)

            grid_size = self._get_grid_size(results[0].shape[0])
            save_path = os.path.join(self._save_dir, 
                    str(cur_global_ind) + '_' + self._overlay_prefix + '.png')
            save_merge_images(np.squeeze(overlay_im_list), 
                                [grid_size, grid_size], save_path, 
                                tanh=self._tanh, color=False)
            cur_global_ind += 1
        else:
            for im_1, im_2 in zip(results[0], results[1]):
                overlay_im = image_overlay(im_1, im_2, color=self._color)
                save_path = os.path.join(self._save_dir, 
                    str(cur_global_ind) + '_' + self._overlay_prefix + '.png')
                scipy.misc.imsave(save_path, np.squeeze(overlay_im))
                cur_global_ind += 1
        self._global_ind = cur_global_ind

class PredictionScalar(PredictionBase):
    def __init__(self, prediction_scalar_tensors, print_prefix):
        """
        Args:
            prediction_scalar_tensors (list): a list of tensor names
            print_prefix (list): a list of name prefix for printing 
                                each tensor in prediction_scalar_tensors
        """

        super(PredictionScalar, self).__init__(prediction_tensors=prediction_scalar_tensors, 
                                             save_prefix=print_prefix)

    def _save_prediction(self, results):
        for re, prefix in zip(results, self._prefix_list):
            print('{} = {}'.format(prefix, re))

class PredictionMeanScalar(PredictionScalar):
    def __init__(self, prediction_scalar_tensors, print_prefix):

        super(PredictionMeanScalar, self).__init__(prediction_scalar_tensors=prediction_scalar_tensors, 
                                             print_prefix=print_prefix)

        self.scalar_list = [[] for i in range(0, len(self._predictions))]

    def _save_prediction(self, results):
        cnt = 0
        for re, prefix in zip(results, self._prefix_list):
            print('{} = {}'.format(prefix, re))
            self.scalar_list[cnt].append(re)
            cnt += 1

    def _after_finish_predict(self):
        for i, prefix in enumerate(self._prefix_list):
            print('Overall {} = {}'.format(prefix, np.mean(self.scalar_list[i])))


class PredictionMat(PredictionBase):
    def _save_prediction(self, results):
        save_path = os.path.join(self._save_dir, 
                               str(self._global_ind) + '_' + 'batch_test' + '.mat')
        scipy.io.savemat(save_path, {name: np.squeeze(val) for name, val 
              in zip(self._prefix_list, results)})

        self._global_ind += 1




        

 


    






