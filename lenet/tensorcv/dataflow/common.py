# File: common.py
# Author: Qian Ge <geqian1001@gmail.com>

import os

from scipy import misc
import numpy as np 

from .preprocess import resize_image_with_smallest_side, random_crop_to_size
from .normalization import identity


def get_file_list(file_dir, file_ext, sub_name=None):
    # assert file_ext in ['.mat', '.png', '.jpg', '.jpeg']
    re_list = []

    if sub_name is None:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.lower().endswith(file_ext)])
    else:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.lower().endswith(file_ext) and sub_name.lower() in name.lower()])
    # for root, dirs, files in os.walk(file_dir):
    #     for name in files:
    #         if name.lower().endswith(file_ext):
    #             re_list.append(os.path.join(root, name))
    # return np.array(re_list)

def get_folder_list(folder_dir):
    return np.array([os.path.join(folder_dir, folder) 
                    for folder in os.listdir(folder_dir) 
                    if os.path.join(folder_dir, folder)]) 

def get_folder_names(folder_dir):
    return np.array([name for name in os.listdir(folder_dir) 
                    if os.path.join(folder_dir, name)])    

def input_val_range(in_mat):
    # TODO to be modified    
    max_val = np.amax(in_mat)
    min_val = np.amin(in_mat)
    if max_val > 1:
        max_in_val = 255.0
        half_in_val = 128.0
    elif min_val >= 0:
        max_in_val = 1.0
        half_in_val = 0.5
    else:
        max_in_val = 1.0
        half_in_val = 0
    return max_in_val, half_in_val

def tanh_normalization(data, half_in_val):
    return (data*1.0 - half_in_val)/half_in_val


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot

def reverse_label_dict(label_dict):
    label_dict_reverse = {}
    for key, value in label_dict.items():
        label_dict_reverse[value] = key
    return label_dict_reverse

def load_image(im_path, read_channel=None, pf=identity, resize=None, resize_crop=None):
    if resize is not None:
        print_warning('[load_image] resize will be unused in the future!\
                      Use pf (preprocess_fnc) instead.')
    if resize_crop is not None:
        print_warning('[load_image] resize_crop will be unused in the future!\
                      Use pf (preprocess_fnc) instead.')

    # im = cv2.imread(im_path, self._cv_read)
    if read_channel is None:
        im = misc.imread(im_path)
    elif read_channel == 3:
        im = misc.imread(im_path, mode='RGB')
    else:
        im = misc.imread(im_path, flatten=True)

    if len(im.shape) < 3:
        try:
            im = misc.imresize(im, (resize[0], resize[1], 1))
        except TypeError:
            pass
        if resize_crop is not None:
            im = resize_image_with_smallest_side(im, resize_crop)
            im = random_crop_to_size(im, resize_crop)
        im = pf(im)
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    else:
        try:
            im = misc.imresize(im, (resize[0], resize[1], im.shape[2]))
        except TypeError:
            pass
        if resize_crop is not None:
            im = resize_image_with_smallest_side(im, resize_crop)
            im = random_crop_to_size(im, resize_crop)
        im = pf(im)
        im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im


def print_warning(warning_str):
    print('[**** warning ****] {}'.format(warning_str))

