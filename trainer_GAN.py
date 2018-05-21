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
    decay_every = 1000
    beta1 = 0.5


    # Parameters for which generator / discriminator to use
    gen_scope = 'generator'
    disc_scope = 'discriminator'
    #gen_scope = 'generator_res'
    #disc_scope = 'discriminator_res'


    # Define placeholders for image and text data
    #caption_generator = tf.placeholder('float32', [batch_size, 201, 70], name='text_generator')
    #caption_right = tf.placeholder('float32', [batch_size, 201, 70], name='encoded_right_text')
    #caption_wrong = tf.placeholder('float32', [batch_size, 201, 70], name='encoded_wrong_text')

    #real_image = tf.placeholder('float32', [batch_size, 64, 64, 3], name='real_image')

    z = tf.placeholder('float32', [conf.GAN_BATCH_SIZE, 100], name='noise')


    # Encoded texts
    #text_G = tf.placeholder('float32', [batch_size, 1024], name='text_generator')
    #text_right = tf.placeholder('float32', [batch_size, 1024], name='encoded_right_text')
    #text_wrong = tf.placeholder('float32', [batch_size, 1024], name='encoded_wrong_text')

    datasource = GanDataLoader()
    iterator, next, (label, text_right, real_image) = datasource.correct_pipe()
    iterator_incorrect, next_incorrect, (label2, text_wrong, real_image2) = datasource.incorrect_pipe()
    iterator_txt_G, next_txt_G, (label3, text_G, real_image_G) = datasource.text_only_pipe()


    # text_G = build_char_cnn_rnn(caption_generator)
    # text_right = build_char_cnn_rnn(caption_right)
    # text_wrong = build_char_cnn_rnn(caption_wrong)


    # Outputs from G and D
    fake_image = generator(text_G, z)
    S_r, debug_1 = discriminator(real_image, text_right)
    S_w, debug_2 = discriminator(real_image2, text_wrong) # todo: maybe here take real_image2
    S_f, debug_3 = discriminator(fake_image, text_G)


    # Loss functions for G and D
    G_loss = -tf.reduce_mean(tf.log(S_f))
    D_loss = -tf.reduce_mean(tf.log(S_r) + (tf.log(1 - S_w) + tf.log(1 - S_f))/2)
    tf.summary.scalar('generator_loss', G_loss)
    tf.summary.scalar('discriminator_loss', D_loss)


    # Parameters we want to train, and their gradients
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scope)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=disc_scope)
    G_grads = tf.gradients(G_loss, G_vars)
    D_grads = tf.gradients(D_loss, D_vars)


    # This is to be able to change the learning rate while training
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)


    # Optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_v, beta1=beta1)
    G_opt = optimizer.apply_gradients(zip(G_grads, G_vars))
    D_opt = optimizer.apply_gradients(zip(D_grads, D_vars))

    #saver = tf.train.Saver()

    # Write to tensorboard
    merged = tf.summary.merge_all()
    fake_img_summary_op = tf.summary.image('generated_image', tf.concat([fake_image * 127.5, real_image_G * 127.5], axis=2))
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p_GAN")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())

    # Execute the graph
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # Yonk stuff here

        encoder_saver = tf.train.import_meta_graph('assets/char-rnn-cnn-19999.meta')
        encoder_saver.restore(sess, 'assets/char-rnn-cnn-19999')

        sess.run(iterator.initializer)
        sess.run(iterator_incorrect.initializer)
        sess.run(iterator_txt_G.initializer)

        # end of yonk's stuff

        step = 0

        # dl = DataLoader()
        # img_r = crop_and_flip(imread('assets/encoder_train/images/image_06734.jpg'),64,
        #                            crop_just_one=True).reshape([-1, 64, 64, 3])
        #
        # plt.imshow(img_r[0])
        # plt.title('before normalizing')
        # plt.show()
        # img_r = (img_r - 127.5)/127.5 #normalize_images(img_r)
        # plt.imshow(img_r[0])
        # plt.title('normalized image')
        # plt.show()
        #
        # f_r = open('assets/encoder_train/captions/class_00001/image_06734.txt')
        # txt_right = f_r.readlines()[0].strip()
        # caption_r = dl._onehot_encode_text(txt_right).reshape([batch_size, 201, 70])
        #
        # f_w = open('assets/encoder_train/captions/class_00018/image_04244.txt')
        # txt_wrong = f_w.readlines()[2].strip()
        # caption_w = dl._onehot_encode_text(txt_wrong).reshape([batch_size, 201, 70])
        #
        # caption_g = caption_r.copy()


        # todo: give me a pipeline yonk the guru plumber
        #img_r, caption_r, caption_w, caption_g = loaddata()


        for epoch in range(epochs):

            # Updating the learning rate every 100 epochs (starting after first 1000 update steps)
            if epoch != 0 and step > 1000 and (epoch % decay_every == 0):
                sess.run(tf.assign(lr_v, lr_v * lr_decay))
                log = " ** new learning rate: %f" % (lr * lr_decay)
                print(log)


            # todo: add condition for mini batches
            step += 1


            # Sample noise, and synthesize image with generator
            z_sample = np.random.normal(0, 1, (conf.GAN_BATCH_SIZE, 100)) # apparently better to sample like this than with tf
            #z_sample = tf.random_normal((conf.GAN_BATCH_SIZE, 100))


            # G_feed = {caption_generator: caption_g, z: z_sample}
            #img_f = sess.run(fake_image, feed_dict=G_feed)
            #
            #
            # # Todo: make sure to put the right images. different or same?
            # # feed_dicts for the discriminator
            # feed_rr = {real_image: img_r, caption_right: caption_r}
            # feed_rw = {real_image: img_r, caption_wrong: caption_w}
            # feed_fr = {fake_image: img_f, caption_generator: caption_g} # the only thing that needs to be feed to get the G_loss
            #
            # feed_dict = {**feed_rr, **feed_rw, **feed_fr, z: z_sample} # merge these dictionaries


            # Updates parameters in G and D

            # if step % 3 == 0:
            s_r, s_w, s_f, summary, dloss, gloss, _, _,  fake_img_summary = sess.run([S_r, S_w, S_f, merged, D_loss, G_loss, D_opt, G_opt,fake_img_summary_op], feed_dict={z:z_sample})
            #
            # else:
            # s_r, s_w, s_f, summary, gloss, _,  fake_img_summary = sess.run(
            #     [S_r, S_w, S_f, merged, G_loss, G_opt, fake_img_summary_op], feed_dict={z: z_sample})

            #gloss, _ = sess.run([G_loss, G_opt], feed_dict=feed_fr)


            # Tensorboard stuff
            writer.add_summary(summary, step)
            print('Update: ', step)
            try:
                print('Discriminator loss: ', dloss)
            except:
                aaaaa = 0
            print('Generator loss: ', gloss)

            if epoch % 10 == 0:
                writer.add_summary(fake_img_summary, epoch)

            # if step % 1000 == 0 or epoch == epochs-1:
            #     saver.save(sess, 'saved/', global_step=step)

            # Uncomment to plot synthesized images # TODO Uncomment this for Google Cloud
            # im_plot = 0.5*img_f[0] + 0.5
            # plt.imshow(im_plot)
            # plt.show()

    # Close writer when done training
    writer.close()



# So that we only run things if it's run directly, not if it's imported
if __name__ == '__main__':
    main()
