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
    z = tf.placeholder('float32', [conf.GAN_BATCH_SIZE, 100], name='noise')


    # Encoded texts fed from the pipeline
    datasource = GanDataLoader()
    iterator, next, (label, text_right, real_image) = datasource.correct_pipe()
    iterator_incorrect, next_incorrect, (label2, text_wrong, real_image2) = datasource.incorrect_pipe()
    iterator_txt_G, next_txt_G, (label3, text_G, real_image_G) = datasource.text_only_pipe()


    # This is to be able to change the learning rate while training
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)


    # Optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_v, beta1=beta1)

    # Runs on GPU
    G0_grads_vars, D0_grads_vars, G0_loss, D0_loss = loss_tower(0, optimizer, text_G, real_image, text_right, real_image2, text_wrong)
    G1_grads_vars, D1_grads_vars, G1_loss, D1_loss = loss_tower(1, optimizer, text_G, real_image, text_right, real_image2, text_wrong)


    # extract vars
    G_grads_total = []
    ls = G0_grads_vars + G1_grads_vars
    G_grads, G_vars = [ l[0] for l in ls],[l[1] for l in ls]


    ls = D0_grads_vars + D1_grads_vars
    D_grads, D_vars = [ l[0] for l in ls],[l[1] for l in ls]
    #D_grads = tf.add(D_grads_total)

    G_loss = tf.add(G0_loss, G1_loss)
    D_loss = tf.add(D0_loss, D1_loss)

    # # Outputs from G and D
    # fake_image = generator_resnet(text_G, z)
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
    # G_grads = tf.gradients(G_loss, G_vars)
    # D_grads = tf.gradients(D_loss, D_vars)

    G_opt = optimizer.apply_gradients(zip(G_grads, G_vars))
    D_opt = optimizer.apply_gradients(zip(D_grads, D_vars))


    # Write to tensorboard
    merged = tf.summary.merge_all()
    #fake_img_summary_op = tf.summary.image('generated_image', tf.concat([fake_image * 127.5, real_image_G * 127.5], axis=2))
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p_GAN")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())

    #
    # Execute the graph
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('assets/char-rnn-cnn-19999.meta')
        saver.restore(sess, 'assets/char-rnn-cnn-19999')
        print('restored')

        # Run the initializers for the pipeline
        sess.run([iterator.initializer, iterator_incorrect.initializer, iterator_txt_G.initializer])

        for step in range(epochs):

            # Updating the learning rate every 100 epochs (starting after first 1000 update steps)
            if step != 0 and step > 10000 and (step % decay_every == 0):
                sess.run(tf.assign(lr_v, lr_v * lr_decay))
                log = " ** new learning rate: %f" % (lr * lr_decay)
                print(log)


            # Sample noise, and synthesize image with generator
            z_sample = np.random.normal(0, 1, (conf.GAN_BATCH_SIZE, 100))


            # Updates parameters in G and D, only every third time for D
            print('Update: ', step)
            if step % 1 == 0:
                summary, dloss, gloss, _, _ = sess.run(
                    [merged, D_loss, G_loss, D_opt, G_opt],
                    feed_dict={z:z_sample})

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


    # Close writer when done training
    writer.close()

def loss_tower(gpu_num, optimizer, text_G, real_image, text_right, real_image2, text_wrong):
    # Outputs from G and D
    with tf.device('/gpu:%d' % gpu_num):
        with tf.name_scope('gpu_%d' % gpu_num):
            fake_image = generator_resnet(text_G)
            S_r = discriminator_resnet(real_image, text_right)
            S_w = discriminator_resnet(real_image2, text_wrong)
            S_f = discriminator_resnet(fake_image, text_G)


            # Loss functions for G and D
            G_loss = -tf.reduce_mean(tf.log(S_f), name='G_loss_gpu%d' % gpu_num)
            D_loss = -tf.reduce_mean(tf.log(S_r) + (tf.log(1 - S_w) + tf.log(1 - S_f))/2, name='G_loss_gpu%d' % gpu_num)

            tf.summary.scalar('generator_loss', G_loss, family='GAN')
            tf.summary.scalar('discriminator_loss', D_loss, family='GAN')


            # Parameters we want to train, and their gradients
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            G_grads_vars = optimizer.compute_gradients(G_loss, G_vars)
            D_grads_vars = optimizer.compute_gradients(D_loss, D_vars)

    return G_grads_vars, D_grads_vars, G_loss, D_loss

def plt_image(datasource):
    # Test pipe setup
    # Determinsitic
    iter_test0, next_test0, (label4, test_deter_txt, test_deter_img) = datasource.test_pipe(deterministic=True)
    test_deter_G_img = generator_resnet(test_deter_txt, z, z_size=1)
    img = tf.concat([test_deter_G_img * 127.5, test_deter_img * 127.5], axis=2)
    test_deter_summary_op = tf.summary.image('test_deterministic',img, family='teststuff')

    # Underterministic
    iter_test1, next_test1, (label4, test_undeter_txt, test_undeter_img) = datasource.test_pipe(deterministic=False)
    test_undeter_G_img = generator_resnet(test_undeter_txt, z, z_size=1)
    img = tf.concat([test_undeter_G_img * 127.5, test_undeter_img * 127.5], axis=2)
    test_undeter_summary_op = tf.summary.image('test_non_determ',img, family='teststuff')

    test_batch_txt = tf.concat([test_deter_txt, test_undeter_txt],axis=0)
    test_batch_img = tf.concat([test_deter_img, test_undeter_img], axis=0)
    test_batch_G_img = generator_resnet(test_batch_txt, z, z_size=2)
    img = tf.concat([test_batch_G_img * 127.5, test_batch_img * 127.5], axis=2)
    test_batch_summary_op = tf.summary.image('test_batch',img, family='teststuff')

    #Below belongs to session

    if step % 100 == 0:
        sess.run([iter_test0.initializer, iter_test1.initializer])
        test_deter, test_undeter, test_batch_summary = sess.run(
            [test_deter_summary_op, test_undeter_summary_op, test_batch_summary_op])
        writer.add_summary(test_deter, step)
        writer.add_summary(test_undeter, step)
        writer.add_summary(test_batch_summary, step)




# So that we only run things if it's run directly, not if it's imported
if __name__ == '__main__':
    main()
