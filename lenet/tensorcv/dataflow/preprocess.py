#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: preprocess.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from scipy import misc


def image_fliplr(image):
    """ Generate mirror image

    Args:
        image (np.array): a 2-D image of shape
            [height, width] or a 3-D image of shape
            [height, width, channels].

    Returns:
        mirror version of original image.
    """
    return np.fliplr(image)


def resize_image_with_smallest_side(image, small_size):
    """
    Resize single image array with smallest side = small_size and
    keep the original aspect ratio.

    Args:
        image (np.array): a 2-D image of shape
            [height, width] or a 3-D image of shape
            [height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.

    Returns:
        rescaled image
    """
    im_shape = image.shape
    shape_dim = len(im_shape)
    assert shape_dim <= 3 and shape_dim >= 2,\
        'Wrong format of image!Shape is {}'.format(im_shape)

    height = float(im_shape[0])
    width = float(im_shape[1])

    if height <= width:
        new_height = int(small_size)
        new_width = int(width * new_height/height)
    else:
        new_width = int(small_size)
        new_height = int(height * new_width/width)

    if shape_dim == 2:
        im = misc.imresize(image, (new_height, new_width))
    elif shape_dim == 3:
        im = misc.imresize(image, (new_height, new_width, image.shape[2]))
    return im


def random_crop_to_size(image, crop_size):
    """ Rondomly crop an image into crop_size

    Args:
        image (np.array): a 2-D image of shape
            [height, width] or a 3-D image of shape
            [height, width, channels].
            The size has to be larger than cropped image.
        crop_size (int or length 2 list): The image size after cropped.

    Returns:
        cropped image
    """
    crop_size = get_shape2D(crop_size)
    im_shape = image.shape
    shape_dim = len(im_shape)
    assert shape_dim <= 3 and shape_dim >= 2, 'Wrong format of image!'

    height = im_shape[0]
    width = im_shape[1]
    assert height >= crop_size[0] and width >= crop_size[1],\
        'Image must be larger than crop size! {}'.format(im_shape)

    s_h = int(np.floor((height - crop_size[0] + 1) * np.random.rand()))
    s_w = int(np.floor((width - crop_size[1] + 1) * np.random.rand()))

    return image[s_h:s_h + crop_size[0], s_w:s_w + crop_size[1]]


def four_connor_crop(image, crop_size):
    """ Crop an image into crop_size with four corner crops

    Args:
        image (np.array): a 2-D image of shape
            [height, width] or a 3-D image of shape
            [height, width, channels].
            The size has to be larger than cropped image.
        crop_size (int or length 2 list): The image size after cropped.

    Returns:
        four cropped images
    """
    crop_size = get_shape2D(crop_size)
    im_shape = image.shape
    shape_dim = len(im_shape)
    assert shape_dim <= 3 and shape_dim >= 2, 'Wrong format of image!'
    height = im_shape[0]
    width = im_shape[1]
    assert height >= crop_size[0] and width >= crop_size[1],\
        'Image must be larger than crop size! {}'.format(im_shape)

    crop_im = []
    crop_im.append(image[: crop_size[0], : crop_size[1]])
    crop_im.append(image[: crop_size[0], width - crop_size[1]:])
    crop_im.append(image[height - crop_size[0]:, : crop_size[1]])
    crop_im.append(image[height - crop_size[0]:, width - crop_size[1]:])

    return crop_im


def center_crop(image, crop_size):
    """ Center crop an image into crop_size

    Args:
        image (np.array): a 2-D image of shape
            [height, width] or a 3-D image of shape
            [height, width, channels].
            The size has to be larger than cropped image.
        crop_size (int or length 2 list): The image size after cropped.

    Returns:
        cropped images
    """
    crop_size = get_shape2D(crop_size)
    im_shape = image.shape
    shape_dim = len(im_shape)
    assert shape_dim <= 3 and shape_dim >= 2, 'Wrong format of image!'
    height = im_shape[0]
    width = im_shape[1]
    assert height >= crop_size[0] and width >= crop_size[1],\
        'Image must be larger than crop size! {}'.format(im_shape)

    return image[(height - crop_size[0])//2:(height + crop_size[0])//2,
                 (width - crop_size[1])//2:(width + crop_size[1])//2]


def random_mirror_resize_crop(image, crop_size, scale_range, mirror_rate=0.5):
    """ Ramdomly rescale, crop and image.

    Args:
        image (np.array): a 2-D image of shape
            [height, width] or a 3-D image of shape
            [height, width, channels].
        crop_size (int or length 2 list): The image size after cropped.
        scale_range (list of int with length 2): The range of scale.
        mirror_rate (float): The probability of mirror image.
            Must within the range [0, 1]

    Returns:
        cropped and rescaled images
    """
    im_shape = image.shape
    shape_dim = len(im_shape)

    assert mirror_rate >= 0 and mirror_rate <= 1,\
        'mirror rate must be in range of [0, 1]!'
    assert shape_dim <= 3 and shape_dim >= 2,\
        'Wrong format of image!Shape is {}'.format(im_shape)

    small_size = int(np.random.rand() * (max(scale_range) - min(scale_range))
                     + min(scale_range))
    image = resize_image_with_smallest_side(image, small_size)

    image = random_crop_to_size(image, crop_size)

    if np.random.rand() >= mirror_rate:
        image = image_fliplr(image)

    return image


def get_shape2D(in_val):
    """
    Return a 2D shape 

    Args:
        in_val (int or list with length 2) 

    Returns:
        list with length 2
    """
    if in_val is None:
        return None
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))
