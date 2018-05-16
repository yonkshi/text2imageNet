import tensorflow as tf
import numpy as np

def loss(V, T):

    """
    Inputs come as a minibatch, disjoint classes!
    :param V: Batch of encoded images. n x 1024
    :param T: Batch of encoded texts. n x 1024
    :return: Loss of the batch
    """

    ########## TF vectorized ##########

    score = tf.matmul(V, tf.matrix_transpose(T))
    thresh = tf.nn.relu(score - tf.diag(score) + 1)
    loss = tf.reduce_mean(thresh)

    return loss


# batch size and dimensionality
n = 40
d = 1024

# Define the graph
V = tf.constant(np.random.normal(0, 1, (n, d)), dtype=tf.float32, shape=(n, d))
T = tf.constant(np.random.normal(0, 1, (n, d)), dtype=tf.float32, shape=(n, d))
shape = tf.shape(V)
l = loss(V, T)

# Execute the graph
with tf.Session() as sess:

    l_out, V_out, T_out, shape_out = sess.run([l, V, T, shape])
    a = 0

