import datetime
from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import conf
from utils import *
import tensorflow as tf


def main():
    """The famous main function that no one knows what it's for"""

    # Training parameters
    epochs = 60000
    lr = 0.0002
    lr_decay = 0.5
    decay_every = 10000
    save_every = 1000
    beta1 = 0.5


    # Parameters for which generator / discriminator to use
    gen_scope = 'generator'
    disc_scope = 'discriminator'
    #gen_scope = 'generator_res'
    #disc_scope = 'discriminator_res'


    # Placeholder for the noise that the generator uses to produce a synthesized image
    #z = tf.placeholder('float32', [conf.GAN_TOWER_BATCH_SIZE, 100], name='noise')


    # Encoded texts fed from the pipeline
    datasource = GanDataLoader()
    text_right, real_image = datasource.correct_pipe()
    text_wrong, real_image2 = datasource.incorrect_pipe()
    text_G, real_image_G = datasource.text_only_pipe()


    # This is to be able to change the learning rate while training
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)


    # Optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_v, beta1=beta1)
    def split_tensor_for_gpu(t):
        return tf.split(t,2)

    # Split for GPU
    text_right0, text_right1 = split_tensor_for_gpu(text_right)
    real_image0, real_image1 = split_tensor_for_gpu(real_image)
    text_wrong0, text_wrong1 = split_tensor_for_gpu(text_wrong)
    real_image20, real_image21 = split_tensor_for_gpu(real_image2)
    text_G0, text_G1 = split_tensor_for_gpu(text_G)
    real_image_G0, real_image_G1 = split_tensor_for_gpu(real_image_G)

    # Runs on GPU
    G0_grads_vars, D0_grads_vars, G0_loss, D0_loss = loss_tower(0, optimizer, text_G0, real_image0, text_right0, real_image20, text_wrong0)
    G1_grads_vars, D1_grads_vars, G1_loss, D1_loss = loss_tower(1, optimizer, text_G1, real_image1, text_right1, real_image21, text_wrong1)

    # # extract vars
    # G_grads_total = []
    # ls = G0_grads_vars + G1_grads_vars
    G_vars = [var for grad, var in G0_grads_vars] #G0 and G1 share vars, so doesn't matter
    D_vars = [var for grad, var in D0_grads_vars] # D0 and D1 share vars, so doesn't matter

    G_grads =[ (g0_grad + g1_grad) / 2 for (g0_grad, g0_vars), (g1_grad, g1_vars) in zip(G0_grads_vars,G1_grads_vars)]
    D_grads = [ (d0_grad + d1_grad) / 2 for (d0_grad, d0_vars), (d1_grad, d1_vars) in zip(D0_grads_vars,D1_grads_vars)]
    G_loss = tf.add(G0_loss, G1_loss)
    D_loss = tf.add(D0_loss, D1_loss)

    # # Single GPU stuff
    # fake_image = generator_resnet(text_G)
    # S_r = discriminator_resnet(real_image, text_right)
    # S_w = discriminator_resnet(real_image2, text_wrong)
    # S_f = discriminator_resnet(fake_image, text_G)
    #
    #
    # # Loss functions for G and D
    # G_loss = -tf.reduce_mean(tf.log(S_f))
    # D_loss = -tf.reduce_mean(tf.log(S_r) + (tf.log(1 - S_w) + tf.log(1 - S_f))/2)
    # tf.summary.scalar('generator_loss', G_loss)
    # tf.summary.scalar('discriminator_loss', D_loss)
    #
    #
    # # Parameters we want to train, and their gradients
    # G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scope)
    # D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=disc_scope)
    #G_grads = tf.gradients(G_loss, G_vars)
    #D_grads = tf.gradients(D_loss, D_vars)
    #

    G_opt = optimizer.apply_gradients(zip(G_grads, G_vars))
    D_opt = optimizer.apply_gradients(zip(D_grads, D_vars))


    # Write to tensorboard
    tf.summary.scalar('generator_loss', G_loss, family='GAN')
    tf.summary.scalar('discriminator_loss', D_loss, family='GAN')
    merged = tf.summary.merge_all()

    #fake_img_summary_op = tf.summary.image('generated_image', tf.concat([fake_image * 127.5, real_image_G * 127.5], axis=2))
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p_GAN")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())

    #
    # Execute the graph
    testset_op = setup_testset(datasource)
    with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True)) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('assets/char-rnn-cnn-19999.meta')
        saver.restore(sess, 'assets/char-rnn-cnn-19999')
        print('restored')

        datasource.preprocess_data_and_initialize(sess)
        # Run the initializers for the pipeline


        for step in range(epochs):

            # Updating the learning rate every 100 epochs (starting after first 1000 update steps)
            if step != 0 and step > 10000 and (step % decay_every == 0):
                sess.run(tf.assign(lr_v, lr_v * lr_decay))
                log = " ** new learning rate: %f" % (lr * lr_decay)
                print(log)


            # Updates parameters in G and D, only every third time for D
            print('Update: ', step)
            if step % 3 == 0:
                summary, dloss, gloss, _, _ = sess.run(
                    [merged, D_loss, G_loss, D_opt, G_opt])

                print('Discriminator loss: ', dloss)
                print('Generator loss: ', gloss)

            else:
                summary, gloss, _ = sess.run(
                    [merged, G_loss, G_opt])

                print('Generator loss: ', gloss)


            # Tensorboard stuff
            writer.add_summary(summary, step)


            # if step % 10 == 0:
            #     writer.add_summary(fake_img_summary, step)

            if step % save_every == 0:
                saver.save(sess, 'saved/', global_step=step)

            if step % 10 == 0:
                testset_op(sess, writer, step)


    # Close writer when done training
    writer.close()

def loss_tower(gpu_num, optimizer, text_G, real_image, text_right, real_image2, text_wrong):
    # Outputs from G and D
    with tf.device('/gpu:%d' % gpu_num):
        with tf.name_scope('gpu_%d' % gpu_num):
            fake_image = generator_resnet(text_G, enable_res=True)
            S_r = discriminator_resnet(real_image, text_right, enable_res=True)
            S_w = discriminator_resnet(real_image2, text_wrong, enable_res=True)
            S_f = discriminator_resnet(fake_image, text_G, enable_res=True)

            # Loss functions for G and D
            G_loss = -tf.reduce_mean(tf.log(S_f), name='G_loss_gpu%d' % gpu_num)
            D_loss = -tf.reduce_mean(tf.log(S_r) + (tf.log(1 - S_w) + tf.log(1 - S_f))/2, name='G_loss_gpu%d' % gpu_num)

            # Parameters we want to train, and their gradients
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            G_grads_vars = optimizer.compute_gradients(G_loss, G_vars)
            D_grads_vars = optimizer.compute_gradients(D_loss, D_vars)

    return G_grads_vars, D_grads_vars, G_loss, D_loss

def setup_testset(datasource):
    # Test pipe setup

    # Non-derterministic
    sample_size = 10
    test_nondeter_txt, test_nondeter_img = datasource.test_pipe(deterministic=False, sample_size = sample_size)

    test_batch_G_img = generator_resnet(test_nondeter_txt,  z_size=sample_size)
    img = tf.concat([test_batch_G_img * 127.5, test_nondeter_img * 127.5], axis=2)
    test_batch_summary_op = tf.summary.image('test_batch', img, family='test_images', max_outputs=10)

    # To be run inside a session
    def testset_op(sess, writer, step):
        test_batch_summary = sess.run(test_batch_summary_op)
        writer.add_summary(test_batch_summary, step)

    return testset_op




# So that we only run things if it's run directly, not if it's imported
if __name__ == '__main__':
    main()
