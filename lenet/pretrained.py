import tensorflow as tf
import numpy as np
import scipy.misc
import argparse

from lenet.tensorcv.dataflow.image import ImageFromFile
from lenet.lib.nets.googlenet import GoogleNet
from lenet.lib.utils.preprocess import resize_image_with_smallest_side, center_crop_image
from lenet.lib.utils.classes import get_word_list



def generated_lenet(images):
    '''
    Loads pretrained GoogleLeNet model and ready to use
    :return: embed_op: output pipe for 1024 dim encoding
             image_placeholder: input pipe for placing processed
             model: entire google lenet model
    '''
    model = GoogleNet(is_load=True, pre_train_path='assets/lenet_pretrained.npy')
    image_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    word_dict = get_word_list('assets/lenet_labels.txt')
    model.create_model([images, 1])
    embed_op = model.layer['embedding']

    return embed_op, image_placeholder


