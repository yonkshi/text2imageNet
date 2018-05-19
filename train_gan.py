from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
import matplotlib.pyplot as plt
import conf
import tensorflow as tf

def main():
    z = tf.random_normal((1, 1, 1,100,))
    x = tf.ones((1, 1, 1, 128,))
    conved = generator(x, z)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(conved)
        print(out)

if __name__ == '__main__':
    main()