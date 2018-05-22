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
    epochs = 600_000
    lr = 0.0002
    lr_decay = 0.5
    decay_every = 100_000
    save_every = 10_000
    beta1 = 0.5
    force_gpu = True

    hp_str = 'Force_gpu:{}\ndecay_every:{}\ndecay_rate:{}\blearning_rate:{}\nepochs:{}\nforce_gpu:{}'.format(force_gpu,decay_every,decay_every,lr,epochs,force_gpu)
    outer_string = tf.convert_to_tensor(hp_str)
    tf.summary.text('configuration', outer_string)


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

    # TODO Double GPU Begin =========
    # Split for GPU
    text_right0, text_right1 = split_tensor_for_gpu(text_right)
    real_image0, real_image1 = split_tensor_for_gpu(real_image)
    text_wrong0, text_wrong1 = split_tensor_for_gpu(text_wrong)
    real_image20, real_image21 = split_tensor_for_gpu(real_image2)
    text_G0, text_G1 = split_tensor_for_gpu(text_G)


    # Runs on GPU
    # G_grads_vars, D_grads_vars, G_loss, D_loss = loss_tower(0, optimizer, text_G, real_image, text_right, real_image2, text_wrong)
    # #G1_grads_vars, D1_grads_vars, G1_loss, D1_loss = loss_tower(1, optimizer, text_G1, real_image1, text_right1, real_image21, text_wrong1)
    #
    # # # extract vars
    # G_vars = [var for grad, var in G0_grads_vars] #G0 and G1 share vars, so doesn't matter
    # D_vars = [var for grad, var in D0_grads_vars] # D0 and D1 share vars, so doesn't matter
    #
    # G_grads =[ (g0_grad + g1_grad) / 2 for (g0_grad, g0_vars), (g1_grad, g1_vars) in zip(G0_grads_vars,G1_grads_vars)]
    # D_grads = [ (d0_grad + d1_grad) / 2 for (d0_grad, d0_vars), (d1_grad, d1_vars) in zip(D0_grads_vars,D1_grads_vars)]
    # G_loss = (G0_loss + G1_loss) / 2
    # D_loss = (D0_loss + D1_loss) /2

    # Single GPU # TODO SINGLE GPU BEGIN ============
    #G_grads_vars, D_grads_vars, G_loss, D_loss = loss_tower(0, optimizer, text_G, real_image, text_right, real_image2,
    #                                                         text_wrong)
    #
    # G_opt = optimizer.apply_gradients(G_grads_vars)
    # D_opt = optimizer.apply_gradients(D_grads_vars)


    # TODO OLD SETUP BEGIN ========

    # Outputs from G and D
    fake_image = generator_resnet(text_G)
    S_r = discriminator_resnet(real_image, text_right)
    S_w = discriminator_resnet(real_image2, text_wrong)
    S_f = discriminator_resnet(fake_image, text_G)


    # Loss functions for G and D
    G_loss = -tf.reduce_mean(tf.log(S_f))
    D_loss = -tf.reduce_mean(tf.log(S_r) + (tf.log(1 - S_w) + tf.log(1 - S_f))/2)
    tf.summary.scalar('generator_loss', G_loss)
    tf.summary.scalar('discriminator_loss', D_loss)



    # Parameters we want to train, and their gradients
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    G_grads = tf.gradients(G_loss, G_vars)
    D_grads = tf.gradients(D_loss, D_vars)

    G_opt = optimizer.apply_gradients(zip(G_grads, G_vars))
    D_opt = optimizer.apply_gradients(zip(D_grads, D_vars))
    # Metrics:
    testset_op = setup_testset(datasource)


    # Write to tensorboard
    setup_accuracy(text_right, real_image, text_wrong, real_image2, text_G)
    tf.summary.scalar('generator_loss', G_loss, family='GAN')
    tf.summary.scalar('discriminator_loss', D_loss, family='GAN')

    # plot weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var, family='GAN_internal')
    for grad in G_grads:
        tf.summary.histogram(var.name + '/gradient', grad, family='internal')
    for grad in D_grads:
        tf.summary.histogram(var.name + '/gradient', grad, family='internal')

    merged = tf.summary.merge_all()

    #fake_img_summary_op = tf.summary.image('generated_image', tf.concat([fake_image * 127.5, real_image_G * 127.5], axis=2))
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p_GAN")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())

    #
    # Execute the graph


    t0 = time()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=(not force_gpu))) as sess: # Allow fall back to CPU

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
            if step % 10 == 0:
                print('Update: ', step)
                summary, dloss, gloss, _, _ = sess.run(
                    [merged, D_loss, G_loss, D_opt, G_opt])

                print('Discriminator loss: ', dloss)
                print('Generator loss: ', gloss)
                # Tensorboard stuff
                writer.add_summary(summary, step)

            else:
                _, _ = sess.run(
                    [D_opt, G_opt])

            if step % save_every == 0:
                saver.save(sess, 'saved/', global_step=step)


            if step % 100 == 0:
                testset_op(sess, writer, step)

            if step % 1000 == 0:
                print('1000 epoch time:', time()-t0)
                t0 = time()


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

            # Parameters we want to train, and their gradients
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            G_grads_vars = optimizer.compute_gradients(G_loss, G_vars)
            D_grads_vars = optimizer.compute_gradients(D_loss, D_vars)

    return G_grads_vars, D_grads_vars, G_loss, D_loss

def setup_accuracy( c1_txt, c1_img, c2_txt, c2_img, cg_txt):

    txt_in = tf.concat([c1_txt,c2_txt,cg_txt], axis=0)
    g_img = generator_resnet(cg_txt)
    img_in = tf.concat([c1_img, c2_img, g_img], axis=0)

    dout = discriminator_resnet(img_in, txt_in)

    dout = tf.reshape(dout, [-1])
    ones = tf.ones_like(dout)
    zeros = tf.zeros_like(dout)
    dout_stepped = tf.where(tf.greater(dout, 0.5),ones,zeros)

    labels = tf.concat([tf.ones([conf.GAN_BATCH_SIZE]), tf.zeros([conf.GAN_BATCH_SIZE]), tf.zeros([conf.GAN_BATCH_SIZE])], axis=0)

    diff = dout_stepped - labels

    c1, c2, cg = tf.split(diff, 3)

    c1_accuracy = (1 - tf.count_nonzero(c1) / conf.GAN_BATCH_SIZE) * 100
    c2_accuracy = (1 - tf.count_nonzero(c2) / conf.GAN_BATCH_SIZE) * 100
    cg_accuracy = (1 - tf.count_nonzero(cg) / conf.GAN_BATCH_SIZE) * 100
    tf.summary.scalar('real_pair_accuracy', c1_accuracy, family='DiscriminatorAccuracy')
    tf.summary.scalar('wrong_pair_accuracy', c2_accuracy, family='DiscriminatorAccuracy')
    tf.summary.scalar('fake_pair_accuracy', cg_accuracy, family='DiscriminatorAccuracy')


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
