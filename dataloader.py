import numpy as np
from scipy import misc
from scipy.ndimage import imread
import scipy
import conf



def load_and_process_image_batch(): # TODO add batch support
    """
    Loads images and preprocess them into 3 channel images
    :param bathces:
    :return: batches of images tensor. [Batchsize, width, height, 3]
    """

    images = []

    im = imread('assets/training/4.png', mode='RGB') # First time for batch
    resized_im = resize_image_with_smallest_side(im)

    images.append(resized_im)

    im = imread('assets/training/4.png', mode='RGB') # First time for batch
    resized_im = resize_image_with_smallest_side(im)
    images.append(resized_im)
    npim = np.array(images)
    return npim

def load_and_process_captions(): # TODO add batch support
    """
    Loads captions and preprocess them into 1 hot encoding
    :param bathces:
    :return:
    """
    # Test string
    test_strings = ["this is a bird", 'this is another bird'] * 20
    return list(map(encode_text, test_strings)) # maps multiple

def encode_text(txt):
    l = list(map(c2i, txt))
    l += [0] * (conf.CHAR_DEPTH - len(l))
    return l

def c2i(c:str):
    return conf.ALPHABET.find(c)
