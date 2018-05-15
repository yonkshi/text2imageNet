#!/usr/bin/env python
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


def intensity_to_rgb(intensity, cmap='jet', normalize=False):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/viz.py>`__.
    Convert a 1-channel matrix of intensities to an RGB image employing
    a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.

    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.

    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """

    # assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    if intensity.ndim == 3:
        return intensity.astype('float32') * 255.0

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def save_merge_images(images, merge_grid, save_path, color=False, tanh=False):
    """Save multiple images with same size into one larger image.

    The best size number is
    int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1

    Args:
        images (np.ndarray): A batch of image array to be merged with size
            [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL].
        merge_grid (list): List of length 2. The grid size for merge images.
        save_path (str): Path for saving the merged image.
        color (bool): Whether convert intensity image to color image.
        tanh (bool): If True, will normalize the image in range [-1, 1]
            to [0, 1] (for GAN models).

    Example:
        The batch_size is 64, then the size is recommended [8, 8].
        The batch_size is 32, then the size is recommended [6, 6].
    """

    # normalization of tanh output
    img = images

    if tanh:
        img = (img + 1.0) / 2.0

    if color:
        # TODO
        img_list = []
        for im in np.squeeze(img):
            im = intensity_to_rgb(np.squeeze(im), normalize=True)
            img_list.append(im)
        img = np.array(img_list)
        # img = np.expand_dims(img, 0)

    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] <= 4):
        img = np.expand_dims(img, 0)
    # img = images
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * merge_grid[0], w * merge_grid[1], 3))
    if len(img.shape) < 4:
        img = np.expand_dims(img, -1)

    for idx, image in enumerate(img):
        i = idx % merge_grid[1]
        j = idx // merge_grid[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :] = image

    scipy.misc.imsave(save_path, merge_img)


def image_overlay(im_1, im_2, color=True, normalize=True):
    """Overlay two images with the same size.

    Args:
        im_1 (np.ndarray): image arrary
        im_2 (np.ndarray): image arrary
        color (bool): Whether convert intensity image to color image.
        normalize (bool): If both color and normalize are True, will
            normalize the intensity so that it has minimum 0 and maximum 1.

    Returns:
        np.ndarray: an overlay image of im_1*0.5 + im_2*0.5
    """
    if color:
        im_1 = intensity_to_rgb(np.squeeze(im_1), normalize=normalize)
        im_2 = intensity_to_rgb(np.squeeze(im_2), normalize=normalize)

    return im_1*0.5 + im_2*0.5
