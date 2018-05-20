import datetime
from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
import matplotlib.pyplot as plt
import conf
import tensorflow as tf

def main():

    ###======================== DEFIINE MODEL ===================================###
    t_caption = tf.placeholder('float32', [None, conf.CHAR_DEPTH, conf.ALPHA_SIZE], name = 'caption_input')

    t_accuracy_caption_mx = tf.placeholder('float32', [None, 1024], name='accuracy_caption_matrix')
    t_accuracy_labels = tf.placeholder('int64', [None], name='accuracy_labels')

    # t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    # t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    # t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    # t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    # Should be 300 maybe
    epochs = 20000
    lr = 0.0007

    # raw input
    data = DataLoader()
    data.process_data()

    # Setting up Queue
    txt_encoder = build_char_cnn_rnn(t_caption)
    lenet_encoded, lenet_image, lenet_model = generated_lenet()

    #lenet_out = tf.stop_gradient(lenet_encoded)

    # Variables we want to train / get gradients for
    txt_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='txt_encode')

    # Loss
    loss = encoder_loss(lenet_encoded, txt_encoder)
    tf.summary.scalar('loss', loss)


    # Gradients. # todo: clip by global norm 5?
    grads = tf.gradients(loss, txt_encoder_vars)

    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    encode_opt = optimizer.apply_gradients(zip(grads, txt_encoder_vars))

    merged = tf.summary.merge_all()

    txt_class_mean = tf.reduce_mean(txt_encoder, axis=0)

    # to save the graph and all variables
    saver = tf.train.Saver()

    # TODO TESTING ACCURAYC, MERGE BACK INTO FUNCTION
    captions_T = tf.transpose(t_accuracy_caption_mx)
    dotted = tf.matmul(lenet_encoded, captions_T)
    predicted = tf.argmax(dotted, axis=1)

    diff = t_accuracy_labels - predicted
    accuracy = tf.scalar_mul(100, 1 - tf.divide(tf.count_nonzero(diff, dtype=tf.int32), tf.size(diff)))
    accuracy_summ = tf.summary.scalar('accuracy', accuracy)

    # Merged summaries for Tensorboard visualization
    #accuracy_summ = tf.summary.merge_all()
    # write to the tensorboard log
    # Saves each run with a unique name
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())

    with tf.Session() as sess:


        sess.run(tf.global_variables_initializer())


        # Yonk test
        caption_mx = []

        for i, sorted_key in enumerate(sorted(data.test_captions.keys())):
            captions = data.test_captions[sorted_key]
            encoded_text_per_class = sess.run(txt_class_mean, feed_dict={t_caption: captions})
            caption_mx.append(encoded_text_per_class)
            print('loaded', i)

        _acc_sum, _acccuracy = sess.run([accuracy_summ, accuracy], feed_dict={t_accuracy_caption_mx: caption_mx,
                                                                              t_accuracy_labels: data.test_labels,
                                                                              lenet_image: data.test_images})
        print('BEFORE_LOADED accuracy: %0.5f' % _acccuracy)

        saver = tf.train.import_meta_graph('assets/char-rnn-cnn-19999.meta')
        saver.restore(sess, 'assets/char-rnn-cnn-19999')

        caption_mx = []
        for i, sorted_key in enumerate(sorted(data.test_captions.keys())):
            captions = data.test_captions[sorted_key]
            encoded_text_per_class = sess.run(txt_class_mean, feed_dict={t_caption: captions})
            caption_mx.append(encoded_text_per_class)
            print('post_loaded', i)

        _acc_sum, _acccuracy = sess.run([accuracy_summ, accuracy], feed_dict={t_accuracy_caption_mx: caption_mx,
                                                                              t_accuracy_labels: data.test_labels,
                                                                              lenet_image: data.test_images})
        print('AFTER_LOADED accuracy: %0.5f' % _acccuracy)

        writer.add_summary(_acc_sum, update)

        for update in range(epochs):

            # Get a mini-batch
            captions, img, txt_seq = data.next_batch()

            dict = {t_caption: txt_seq, lenet_image: img}

            # Update parameters
            sess.run(encode_opt, feed_dict=dict)

            # Calculate the loss
            summary, loss_out, encoded_text, encoded_image = sess.run([merged, loss, txt_encoder, lenet_encoded],
                                                             feed_dict=dict)

                # write to the tensorboard summary
            writer.add_summary(summary, update)
            print('loss: ', loss_out)

            if update % 1000 == 0 or update == epochs-1:
                saver.save(sess, './text_encoder/char-rnn-cnn', global_step=update)

            if update % 100 == 0:
                caption_mx = []
                for sorted_key in sorted(data.test_captions.keys()):
                    captions = data.test_captions[sorted_key]
                    encoded_text_per_class = sess.run(txt_class_mean, feed_dict={t_caption:captions})
                    caption_mx.append(encoded_text_per_class)

                _acc_sum, _acccuracy = sess.run([accuracy_summ, accuracy], feed_dict={t_accuracy_caption_mx: caption_mx, t_accuracy_labels:data.test_labels, lenet_image:data.test_images})
                print('accuracy: %0.5f' % _acccuracy)
                writer.add_summary(_acc_sum, update)



    writer.close()


def encoder_loss(V, T):

    """
    Inputs come as a minibatch, disjoint classes!
    :param V: Batch of encoded images. n x 1024
    :param T: Batch of encoded texts. n x 1024
    :return: Loss of the batch
    """

    ########## TF vectorized ##########
    with tf.variable_scope('Loss'):

        n = tf.shape(V)[0]

        score = tf.matmul(V, tf.matrix_transpose(T))
        diag = tf.diag_part(score)
        temp = tf.nn.relu(score - tf.reshape(diag, [-1, 1]) + 1 - tf.eye(n))
        loss = tf.reduce_mean(temp)

        return loss


def encoder_accuracy(labels:tf.Tensor, images:tf.Tensor, captions:tf.Tensor):
    captions_T = tf.transpose(captions)
    dotted = tf.matmul(images, captions_T)
    maxed = tf.argmax(dotted, axis=0)
    pass

# # batch size and dimensionality
# n = 40
# d = 1024
#
# # Define the graph
# V = tf.constant(np.random.normal(0, 1, (n, d)), dtype=tf.float32, shape=(n, d))
# T = tf.constant(np.random.normal(0, 1, (n, d)), dtype=tf.float32, shape=(n, d))
# shape = tf.shape(V)
# l = loss(V, T)
#
# # Execute the graph
# with tf.Session() as sess:
#
#     l_out, V_out, T_out, shape_out = sess.run([l, V, T, shape])
#     a = 0


if __name__ == '__main__':
    main()