from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
import conf
import tensorflow as tf

def main():

    ###======================== DEFIINE MODEL ===================================###
    t_caption = tf.placeholder('float32', [conf.BATCH_SIZE, conf.CHAR_DEPTH, conf.ALPHA_SIZE], name = 'caption_input')
    # t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    # t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    # t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    # t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    # Should be 300 maybe
    epochs = 1000
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


    # Merged summaries for Tensorboard visualization
    merged = tf.summary.merge_all()

    # write to the tensorboard log
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

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

            print(loss_out)


def encoder_loss(V, T):

    """
    Inputs come as a minibatch, disjoint classes!
    :param V: Batch of encoded images. n x 1024
    :param T: Batch of encoded texts. n x 1024
    :return: Loss of the batch
    """

    ########## TF vectorized ##########
    with tf.variable_scope('Loss'):

        score = tf.matmul(V, tf.matrix_transpose(T))
        thresh = tf.nn.relu(score - tf.diag(score) + 1)
        loss = tf.reduce_mean(thresh)

        return loss


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