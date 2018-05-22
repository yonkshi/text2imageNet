import numpy as np
from scipy import misc
from scipy.ndimage import imread
from random import shuffle
import random
import matplotlib.pyplot as plt


def normalize_images(images):

    """
    Normalizing a batch of images so that they have mean (0,0) and std = 1
    :param images: batch_size x 64 x 64 x 3
    :return:
    """

    batch_size, s, s, c = images.shape
    channel_mean = np.einsum('nijk -> k', images) / (batch_size * s * s)
    ims = images - channel_mean
    channel_sd = np.sqrt(np.einsum('nijk -> k', ims**2) / (batch_size * s * s))
    images = ims / channel_sd

    return images

def sample_image_crop_flip(image, output_size=64, scale_size=70, deterministic=False, return_multiple=False):

    im = resize_image_with_smallest_side(image, scale_size)

    h, w, c = im.shape
    os = output_size

    def crop_middle(im): return im[(h - os) // 2:(h + os) // 2, (w - os) // 2:(w + os) // 2, :]
    def crop_topleft(im): return im[:os, :os, :]
    def crop_topright(im): return im[:os, w-os:, :]
    def crop_bottomright(im): return im[h-os:, w-os:, :]
    def crop_bottomleft(im): return im[h-os:, :os, :]

    def flip(im): return np.fliplr(im)
    def donothing(im): return im

    crop = random.choice([crop_middle,crop_topleft,crop_topright,crop_bottomright,crop_bottomleft])
    flip_maybe = random.choice([flip,donothing])

    if return_multiple:
        images = []
        for crop_op in [crop_middle,crop_topright]:
            cropped = crop_op(im)
            images.append(cropped)
            images.append(flip(cropped))
        return np.array(images)

    if deterministic: return crop_middle(im)



    return flip_maybe(crop(im))

def crop_and_flip(image,os=224, scales = [256],crop_just_one=False):

    """
    :param image: An image on tensor form, h x w x 3
    :param size: output
    :return:
    """

    h, w, c = image.shape

    #scales = [80]



    images = []
    for l in scales:
        im = resize_image_with_smallest_side(image, l)

        h, w, c = im.shape

        # crop middle
        im_middle = im[(h - os) // 2:(h + os) // 2, (w - os) // 2:(w + os) // 2, :]

        if crop_just_one:
            return im_middle

        else:
            images.append(im_middle)
            images.append(np.fliplr(im_middle))

            im_upperleft = im[:os, :os, :]
            images.append(im_upperleft)
            images.append(np.fliplr(im_upperleft))

            im_upperright = im[:os, w-os:, :]
            images.append(im_upperright)
            images.append(np.fliplr(im_upperright))

            im_lowerleft = im[h-os:, :os, :]
            images.append(im_lowerleft)
            images.append(np.fliplr(im_lowerleft))

            im_lowerright = im[h-os:, w-os:, :]
            images.append(im_lowerright)
            images.append(np.fliplr(im_lowerright))


    #shuffle(images)

    return images


def resize_image_with_smallest_side(image, small_size=224):
    """
    Resize single image array with smallest side = small_size and
    keep the original aspect ratio.

    Author: Qian Ge <geqian1001@gmail.com>

    Args:
        image (np.array): 2-D image of shape
            [height, width] or 3-D image of shape
            [height, width, channels] or 4-D of shape
            [1, height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.
    """
    im_shape = image.shape
    shape_dim = len(im_shape)
    assert shape_dim <= 4 and shape_dim >= 2,\
        'Wrong format of image!Shape is {}'.format(im_shape)

    if shape_dim == 4:
        image = np.squeeze(image, axis=0)
        height = float(im_shape[1])
        width = float(im_shape[2])
    else:
        height = float(im_shape[0])
        width = float(im_shape[1])

    if height <= width:
        new_height = int(small_size)
        new_width = int(new_height/height * width)
    else:
        new_width = int(small_size)
        new_height = int(new_width/width * height)

    if shape_dim == 2:
        im = misc.imresize(image, (new_height, new_width))
    elif shape_dim == 3:
        im = misc.imresize(image, (new_height, new_width, image.shape[2]))
    else:
        im = misc.imresize(image, (new_height, new_width, im_shape[3]))
        im = np.expand_dims(im, axis=0)

    return im


'''output_size=224
#image = imread('implementation/result.png', mode='RGB')
image = imread('assets/image_00001.jpg', mode='RGB')
plt.imshow(image)
plt.show()
images=crop_and_flip(image,output_size)
for image in images:
    plt.imshow(image)
    plt.show()'''

