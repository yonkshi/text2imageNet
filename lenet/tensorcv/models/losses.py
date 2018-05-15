# File: losses.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import numpy as np


def GAN_discriminator_loss(d_real, d_fake, name='d_loss'):
    print('---- d_loss -----')
    with tf.name_scope(name):
        d_loss_real = comp_loss_real(d_real)
        d_loss_fake = comp_loss_fake(d_fake)
        return tf.identity(d_loss_real + d_loss_fake, name='result')

def GAN_generator_loss(d_fake, name='g_loss'):
    print('---- g_loss -----')
    with tf.name_scope(name):
        return tf.identity(comp_loss_real(d_fake), name='result')

def comp_loss_fake(discrim_output):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discrim_output, 
                                    labels=tf.zeros_like(discrim_output)))

def comp_loss_real(discrim_output):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discrim_output, 
            labels=tf.ones_like(discrim_output)))

