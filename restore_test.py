import tensorflow as tf


def main():

    saver = tf.train.import_meta_graph('saved/-60.meta')
    print('meta graph imported')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'saved/-60')
        print('variables restored')

        writer = tf.summary.FileWriter(logdir='graphs', graph=sess.graph)


if __name__ == '__main__':
    main()