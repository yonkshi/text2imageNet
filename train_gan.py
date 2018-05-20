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
    txt = tf.random_normal((1, 1, 1, 1024,))

    out, test_out = generator(txt, z)
    out2 = generator2(txt, z)

    D_out = discriminator(out2, txt)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out, out2, D_out = sess.run([out, out2, D_out])
        print(out.shape)
        print(out2.shape)
        print(D_out)

if __name__ == '__main__':
    main()