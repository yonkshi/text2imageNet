import tensorflow as tf
from dataloader import *


def onehot_encode_text(txt):
    axis1 = conf.ALPHA_SIZE
    axis0 = conf.CHAR_DEPTH
    oh = np.zeros((axis0, axis1))
    for i, c in enumerate(txt):
        if i >= conf.CHAR_DEPTH:
            break  # Truncate long text
        char_i = conf.ALPHABET.find(c)
        oh[i, char_i] = 1

    # l = list(map(self._c2i, txt))
    # l += [0] * (conf.CHAR_DEPTH - len(l)) # padding
    return oh

def main():

    #loader = GanDataLoader()
    print('creating new text_encode')
    txt = [onehot_encode_text('hello world')]
    txt2 = np.array(txt, dtype='float32')
    txt_t = tf.convert_to_tensor(txt2)
    tf.set_random_seed(15)
    out = build_char_cnn_rnn(txt_t)




    saver = tf.train.Saver()
    print('meta graph imported')

    graph = tf.get_default_graph()
    trainbles_op = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='txt_encode')


    # -0.003698 to 0.0039
    with tf.Session() as sess:



        #text0 = sess.run(txt_out)
        #trainables00, trainables010 = sess.run([trainbles[0], trainbles[10]])
        #trainables0 = sess.run(trainbles)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'assets/char-rnn-cnn-19999')
        print('variables restored')

        #txt_out = build_char_cnn_rnn(txt_t)
        #text1 = sess.run(txt_out)
        #trainables1,trainables10 = sess.run([trainbles_op[0], trainbles_op[10]])
        trainables = sess.run(trainbles_op)
        encoded = sess.run(out)

        writer = tf.summary.FileWriter(logdir='graphs', graph=sess.graph)


if __name__ == '__main__':
    main()