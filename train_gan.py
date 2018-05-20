from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
import matplotlib.pyplot as plt
import conf
import tensorflow as tf

def main():
    z = tf.random_normal((1, 1, 1,100,))
    x = tf.ones((1, 1, 1, 128,))
    txt = tf.random_normal((1, 1, 1, 1024,))
    conved = generator(x, z)
    conved2 = generator2(txt, z)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(conved)
        out2 = sess.run(conved2)
        print(out.shape)
        print(out2.shape)

if __name__ == '__main__':
    main()