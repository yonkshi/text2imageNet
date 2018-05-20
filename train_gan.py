from __future__ import absolute_import, division, print_function
import tensorflow as tf
#tf.enable_eager_execution()

from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
import matplotlib.pyplot as plt
import conf




def main():
    z = tf.random_normal((1, 1, 1,100,))
    x = tf.ones((1, 1, 1, 128,))
    txt = tf.random_normal((1, 1, 1, 1024,))

    generator_out, test_out = generator(txt, z)
    conved2 = generator2(txt, z)

    d_out = descriminator_resnet(generator_out, txt)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        d_out = sess.run(d_out)
        out2 = sess.run(conved2)
        print(d_out.shape)
        print(out2.shape)


if __name__ == '__main__':
    main()