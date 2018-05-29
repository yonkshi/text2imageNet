from models import *
from dataloader import *
import conf

import tensorflow as tf


def main():
    """The famous main function that no one knows what it's for"""
    run_title = 'DEBUG_RUN' if conf.SIMPLE_RUN else input('Please name this session:')
    # Training parameters
    epochs = 600_000
    lr = 0.00002
    lr_decay = 0.5
    decay_every = 100_000
    save_every = 10_000
    beta1 = 0.5
    force_gpu = conf.FORCE_GPU
    num_gpu = conf.NUM_GPU


    # Encoded texts fed from the pipeline
    if conf.END_TO_END:
        datasource = GanDataLoader_NoEncoder()
    else:
        datasource = GanDataLoader()
    text_right, real_image = datasource.correct_pipe()
    text_wrong, real_image2 = datasource.incorrect_pipe()
    text_G, real_image_G = datasource.text_only_pipe()
    text_encoder(text_G, reuse=False)

    # This is to be able to change the learning rate while training
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    # Optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_v, beta1=beta1)
    def split_tensor_for_gpu(t):
        return tf.split(t,num_gpu)

    # TODO Double GPU Begin =========
    # Split for GPU
    c1_txts = split_tensor_for_gpu(text_right)
    c1_imgs = split_tensor_for_gpu(real_image)
    c2_txts = split_tensor_for_gpu(text_wrong)
    c2_imgs = split_tensor_for_gpu(real_image2)
    cg_txts = split_tensor_for_gpu(text_G)

    G_grads = []
    D_grads = []
    G_loss = 0
    D_loss = 0
    for i in range(num_gpu):
        # Runs on GPU
        G_grads_vars, D_grads_vars, G_loss_gpu, D_loss_gpu = loss_tower(i, optimizer, cg_txts[i], c1_imgs[i], c1_txts[i], c2_imgs[i], c2_txts[i])

        # normalize and element wise add
        if not G_grads:
            G_grads = [g_grad / num_gpu  for g_grad, g_vars in G_grads_vars]
            D_grads = [d_grad / num_gpu for d_grad, d_vars in D_grads_vars]
        else:
            # Element wise add to G_grads collection, G_grads is same size as G_grads_vars' grads
            G_grads = [ g_grad / num_gpu + G_grads[j] for j, (g_grad, g_vars) in enumerate(G_grads_vars)]
            D_grads = [ d_grad / num_gpu + D_grads[j]for j, (d_grad, d_vars) in enumerate(D_grads_vars)]

        G_loss = G_loss_gpu / num_gpu + G_loss
        D_loss = D_loss_gpu / num_gpu + D_loss
    # sum and normalize

    ## extract vars
    #G_vars = [var for grad, var in G_grads_vars]  # G0 and G1 share vars, so doesn't matter
    #D_vars = [var for grad, var in D_grads_vars]  # D0 and D1 share vars, so doesn't matter

    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    if conf.END_TO_END:
        # Encoder_grads_G = G_grads[22:] # TODO Hard coded split between GAN grads and encoder grads
        # G_grads = G_grads[:22]
        # Encoder_grads_D = D_grads[24:]
        # D_grads = D_grads[:24]
        Encode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='txt_encode')
        # Encoder_grads = [(g + d)/2 for g, d in zip(Encoder_grads_G, Encoder_grads_D)]
        #E_opt = optimizer.apply_gradients(zip(Encoder_grads, Encode_vars))
        E_opt = tf.constant(5)
    else:
        Encode_vars = []
        E_opt = tf.constant(5)

    #G_vars += Encode_vars
    D_vars += Encode_vars



    G_opt = optimizer.apply_gradients(zip(G_grads, G_vars))
    D_opt = optimizer.apply_gradients(zip(D_grads, D_vars))


    # Single GPU # TODO SINGLE GPU BEGIN ============
    # G_grads_vars, D_grads_vars, G_loss, D_loss = loss_tower(0, optimizer, text_G, real_image, text_right, real_image2,
    #                                                          text_wrong)


    #G_opt = optimizer.apply_gradients(G_grads_vars)
    #D_opt = optimizer.apply_gradients(D_grads_vars)


    # TODO OLD SETUP BEGIN ========

    # Outputs from G and D
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



    # Parameters we want to train, and their gradients
    # G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    # D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    # G_grads = tf.gradients(G_loss, G_vars)
    # D_grads = tf.gradients(D_loss, D_vars)

    # G_opt = optimizer.apply_gradients(zip(G_grads, G_vars))
    # D_opt = optimizer.apply_gradients(zip(D_grads, D_vars))

    # Metrics:
    testset_op = setup_testset(datasource)

    # Write to tensorboard
    setup_accuracy(text_right, real_image, text_wrong, real_image2, text_G)
    tf.summary.scalar('generator_loss', G_loss, family='GAN')
    tf.summary.scalar('discriminator_loss', D_loss, family='GAN')

    hp_str = 'Force_gpu:{}\ndecay_every:{}\ndecay_rate:{}\blearning_rate:{}\nepochs:{}\nforce_gpu:{}'.format(force_gpu,decay_every,decay_every,lr,epochs,force_gpu)
    outer_string = tf.convert_to_tensor(hp_str)
    tf.summary.text('configuration', outer_string)

    # plot weights
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.name, var, family='GAN_internal')
    # for grad, var in zip(G_grads, G_vars):
    #     tf.summary.histogram(var.name + '/gradient', grad, family='internal')
    # for grad, var in zip(D_grads, D_vars):
    #     tf.summary.histogram(var.name + '/gradient', grad, family='internal')

    merged = tf.summary.merge_all()

    #fake_img_summary_op = tf.summary.image('generated_image', tf.concat([fake_image * 127.5, real_image_G * 127.5], axis=2))
    run_name = run_title + datetime.datetime.now().strftime("May_%d_%I_%M%p_GAN")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=(not force_gpu))) as sess: # Allow fall back to CPU
        #tf.set_random_seed(100)
        sess.run(tf.global_variables_initializer())
        # Restore text encoder
        text_encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='txt_encode')
        # saver = tf.train.Saver(text_encoder_vars)
        # saver.restore(sess, 'assets/char-rnn-cnn-19999')

        datasource.preprocess_data_and_initialize(sess)
        # Run the initializers for the pipeline

        t0 = time()
        for step in range(epochs):

            # Updating the learning rate every 100 epochs (starting after first 1000 update steps)
            if step != 0 and step > 10000 and (step % decay_every == 0):
                sess.run(tf.assign(lr_v, lr_v * lr_decay))
                log = " ** new learning rate: %f" % (lr * lr_decay)
                print(log)

            # Updates parameters in G and D, only every third time for D
            if step % 10 == 0:
                print('Update: ', step)
                summary, dloss, gloss, _, _, _= sess.run(
                    [merged, D_loss, G_loss, D_opt, G_opt, E_opt])

                print('Discriminator loss: ', dloss)
                print('Generator loss: ', gloss)
                print('time:', time() - t0 )
                # Tensorboard stuff
                writer.add_summary(summary, step)
            # if step % 2 == 0:
            #     _, _ = sess.run(
            #         [D_opt, G_opt])
            else:
                _, _,_ = sess.run(
                    [D_opt, G_opt, E_opt])

            if step % save_every == 0:
                #saver.save(sess, 'saved/%s' % run_name, global_step=step)
                pass


            if step % 100 == 0:
                testset_op(sess, writer, step)

            if step % 1000 == 0:
                print('1000 epoch time:', time()-t0)
                t0 = time()

    # Close writer when done training
    writer.close()

def loss_tower(gpu_num, optimizer, text_G, real_image, text_right, real_image2, text_wrong, reuse=True):
    # Outputs from G and D
    with tf.device('/gpu:%d' % gpu_num):
        with tf.name_scope('gpu_%d' % gpu_num):

            if conf.END_TO_END:
                text_right = text_encoder(text_right, reuse)
                text_G = text_encoder(text_G, reuse)
                text_wrong = text_encoder(text_wrong, reuse)
                Encode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='txt_encode')
            else:
                Encode_vars = []

            fake_image = generator_resnet(text_G)
            S_r = discriminator_resnet(real_image, text_right)
            S_w = discriminator_resnet(real_image2, text_wrong)
            S_f = discriminator_resnet(fake_image, text_G)

            # Loss functions for G and D
            G_loss = -tf.reduce_mean(tf.log(S_f), name='G_loss_gpu%d' % gpu_num)
            D_loss = -tf.reduce_mean(tf.log(S_r) + (tf.log(1 - S_w) + tf.log(1 - S_f))/2, name='G_loss_gpu%d' % gpu_num)

            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')



            # Parameters we want to train, and their gradients
            G_grads = optimizer.compute_gradients(G_loss, G_vars)
            D_grads = optimizer.compute_gradients(D_loss, D_vars + Encode_vars) # disable text encoder training on D

    return G_grads, D_grads, G_loss, D_loss

def setup_accuracy( c1_txt, c1_img, c2_txt, c2_img, cg_txt, reuse=True):
    with tf.device('/gpu:0'):
        if conf.END_TO_END:
            c1_txt = text_encoder(c1_txt, reuse)
            c2_txt = text_encoder(c2_txt, reuse)
            cg_txt = text_encoder(cg_txt, reuse)
        txt_in = tf.concat([c1_txt, c2_txt, cg_txt], axis=0)

        g_img = generator_resnet(cg_txt, z_size=conf.GAN_BATCH_SIZE)
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

    if conf.END_TO_END:
        test_nondeter_txt = text_encoder(test_nondeter_txt, reuse=True)

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
