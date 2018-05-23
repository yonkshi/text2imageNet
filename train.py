import datetime
from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
import matplotlib.pyplot as plt
import conf
import tensorflow as tf

t_replacelenet = tf.placeholder('float32', [conf.BATCH_SIZE, 1024], name='image_placeholder_for_test')

def main():

    # Should be 300 maybe
    epochs = 20000
    lr = 0.0002
    force_gpu = conf.FORCE_GPU_TEXT_ENCODER
    num_gpu = conf.NUM_GPU_TXT_ENCODER

    # raw input
    data = DataLoader()
    data.process_data()
    iterator, cls, image_batch, text_batch = data.base_pipe()

    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    #lenet_out = tf.stop_gradient(lenet_encoded)
    grads = []
    loss = 0

    for i in range(num_gpu):
        # Runs on GPU
        grads_gpu, loss_gpu = grad_tower(i, text_batch, image_batch)

        # normalize grads from each GPU
        if not grads:
            grads = [grad / num_gpu for grad in grads_gpu]
        else:
            # Element wise add to G_grads collection, G_grads is same size as G_grads_vars' grads
            grads = [grad / num_gpu + grads[j] for j, grad in enumerate(grads_gpu)]

        loss = loss_gpu / num_gpu + loss

    ## extract vars
    txt_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='txt_encode')

    opt_op = optimizer.apply_gradients(zip(grads, txt_encoder_vars))


    tf.summary.scalar('loss', loss)
    merged_summary_op = tf.summary.merge_all()



    # to save the graph and all variables
    saver = tf.train.Saver()
    accuracy_run = accuracy_calc()

    # Merged summaries for Tensorboard visualization
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=(not force_gpu))) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        for update in range(epochs):

            # grads
            inp = np.random.randn(40,1024)
            summary, _, l =  sess.run([merged_summary_op, opt_op, loss], feed_dict={t_replacelenet:inp})

            writer.add_summary(summary, update)
            print('loss: ', l)

            if update % 1000 == 0 or update == epochs-1:
                saver.save(sess, './text_encoder/%s' % run_name, global_step=update)

            if update % 100 == 0:
                accuracy_run(sess,writer,data,update)
                # caption_mx = []
                # for sorted_key in sorted(data.test_captions.keys()):
                #     captions = data.test_captions[sorted_key]
                #     encoded_text_per_class = sess.run(txt_class_mean, feed_dict={t_caption:captions})
                #     caption_mx.append(encoded_text_per_class)
                #
                # _acc_sum, _acccuracy = sess.run([accuracy_summary_op, accuracy_op], feed_dict={t_caption: caption_mx, t_accuracy_labels:data.test_labels, t_image:data.test_images})
                # print('accuracy: %0.5f' % _acccuracy)
                # writer.add_summary(_acc_sum, update)

    writer.close()

def accuracy_calc():
    t_caption = tf.placeholder('float32', [None, conf.CHAR_DEPTH, conf.ALPHA_SIZE], name='caption_input')

    t_accuracy_caption_mx = tf.placeholder('float32', [None, 1024], name='accuracy_caption_matrix')
    t_accuracy_labels = tf.placeholder('int64', [None], name='accuracy_labels')
    t_image = tf.placeholder('float32', [None, None, None, 3], name='image_placeholder_for_test')



    with tf.device('/gpu:0'):
        # accuracy computation
        txt_encoder = build_char_cnn_rnn(t_caption)
        txt_class_mean = tf.reduce_mean(txt_encoder, axis=0)
        lenet_encoded = t_replacelenet #generated_lenet(t_image)
        captions_T = tf.transpose(t_accuracy_caption_mx)
        dotted = tf.matmul(lenet_encoded, captions_T)
        predicted = tf.argmax(dotted, axis=1)

    diff = t_accuracy_labels - predicted
    accuracy_op = tf.scalar_mul(100, 1 - tf.divide(tf.count_nonzero(diff, dtype=tf.int32), tf.size(diff)))
    accuracy_summ_op = tf.summary.scalar('accuracy', accuracy_op)

    def accuracy_run(sess, writer, data, step):
        caption_mx = []
        for sorted_key in sorted(data.test_captions.keys()):
            captions = data.test_captions[sorted_key]
            encoded_text_per_class = sess.run(txt_class_mean, feed_dict={t_caption: captions})
            caption_mx.append(encoded_text_per_class)

        _acc_sum, _acccuracy = sess.run([accuracy_summ_op, accuracy_op],
                                        feed_dict={t_accuracy_caption_mx: caption_mx, t_accuracy_labels: data.test_labels,
                                                   t_image: data.test_images})
        print('accuracy: %0.5f' % _acccuracy)
        writer.add_summary(_acc_sum, step)

    return accuracy_run

def grad_tower(gpu_num, caption, image):
    # Loss
    # Setting up Queue
    with tf.device('/gpu:%d' % gpu_num):
        with tf.name_scope('scope_gpu_%d' % gpu_num):
            txt_encoder = build_char_cnn_rnn(caption)
            lenet_encoded = t_replacelenet #generated_lenet(image)
            loss = encoder_loss(lenet_encoded, txt_encoder)

            txt_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='txt_encode')
            # Gradients. # todo: clip by global norm 5?
            grads = tf.gradients(loss, txt_encoder_vars)

    return grads, loss


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