import os

import numpy as np 
from scipy.io import loadmat

from ..common import *
from ..normalization import *
from ..image import ImageFromFile

__all__ = ['BSDS500', 'BSDS500HED']

class BSDS500(ImageFromFile):
    def __init__(self, name, 
                 data_dir='', 
                 shuffle=True, 
                 normalize=None,
                 is_mask=False,
                 normalize_fnc=identity,
                 resize=None):

        assert name in ['train', 'test', 'val', 'infer']
        self._load_name = name
        self._is_mask = is_mask

        super(BSDS500, self).__init__('.jpg', 
                                        data_dir=data_dir, 
                                        num_channel=3,
                                        shuffle=shuffle, 
                                        normalize=normalize,
                                        normalize_fnc=normalize_fnc,
                                        resize=resize)

    def _load_file_list(self, _):
        im_dir = os.path.join(self.data_dir, 'images', self._load_name)
        self._im_list = get_file_list(im_dir, '.jpg')

        gt_dir = os.path.join(self.data_dir, 'groundTruth', self._load_name)
        self._gt_list = get_file_list(gt_dir, '.mat')

        # TODO may remove later
        if self._is_mask:
            mask_dir = os.path.join(self.data_dir, 'mask', self._load_name)
            self._mask_list = get_file_list(mask_dir, '.mat')

        if self._shuffle:
            self._suffle_file_list()

    def _load_data(self, start, end):
        input_im_list = []
        input_label_list = []
        if self._is_mask:
            input_mask_list = []

        for k in range(start, end):
            cur_path = self._im_list[k] 
            im = load_image(cur_path, read_channel=self._read_channel,
                            resize=self._resize)
            input_im_list.extend(im)

            gt = loadmat(self._gt_list[k])['groundTruth'][0]
            num_gt = gt.shape[0]
            gt = sum(gt[k]['Boundaries'][0][0] for k in range(num_gt))
            gt = gt.astype('float32')
            gt = gt * 1.0 / np.amax(gt)
            # gt = 1.0*gt/num_gt
            try:
                gt = misc.imresize(gt, (self._resize[0], self._resize[1]))
            except TypeError:
                pass
            gt = np.reshape(gt, [1, gt.shape[0], gt.shape[1]])
            input_label_list.extend(gt)

            if self._is_mask:
                mask = np.reshape(gt, [1, mask.shape[0], mask.shape[1]])
                input_mask_list.extend(mask)

        input_im_list = self._normalize_fnc(np.array(input_im_list), 
                                          self._get_max_in_val(), 
                                          self._get_half_in_val())

        input_label_list = np.array(input_label_list)
        if self._is_mask:
            input_mask_list = np.array(input_mask_list)
            return [input_im_list, input_label_list, input_mask_list]
        else:
            return [input_im_list, input_label_list]

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        self._im_list = self._im_list[idxs]
        self._gt_list = self._gt_list[idxs]
        try:
            self._mask_list = self._mask_list[idxs]
        except AttributeError:
            pass

class BSDS500HED(BSDS500):
    def _load_file_list(self, _):
        im_dir = os.path.join(self.data_dir, 'images', self._load_name)
        self._im_list = get_file_list(im_dir, '.jpg')

        gt_dir = os.path.join(self.data_dir, 'groundTruth', self._load_name)
        self._gt_list = get_file_list(gt_dir, '.png')

        if self._shuffle:
            self._suffle_file_list()

    def _load_data(self, start, end):
        input_im_list = []
        input_label_list = []

        for k in range(start, end):
            im = load_image(self._im_list[k], read_channel=self._read_channel,
                            resize=self._resize)
            input_im_list.extend(im)

            gt = load_image(self._gt_list[k], read_channel=1,
                            resize=self._resize)
            gt = gt * 1.0 / np.amax(gt)

            # gt = loadmat(self._gt_list[k])['groundTruth'][0]
            # num_gt = gt.shape[0]
            # gt = sum(gt[k]['Boundaries'][0][0] for k in range(num_gt))
            # gt = gt.astype('float32')
            # gt = 1.0*gt/num_gt
            # try:
            #     gt = misc.imresize(gt, (self._resize[0], self._resize[1]))
            # except TypeError:
            #     pass
            gt = np.squeeze(gt, axis = -1)
            input_label_list.extend(gt)


        input_im_list = self._normalize_fnc(np.array(input_im_list), 
                                          self._get_max_in_val(), 
                                          self._get_half_in_val())

        input_label_list = np.array(input_label_list)

        return [input_im_list, input_label_list]


if __name__ == '__main__':
    a = BSDS500('val','E:\\GITHUB\\workspace\\CNN\\dataset\\BSR_bsds500\\BSR\\BSDS500\\data\\')
    print(a.next_batch())