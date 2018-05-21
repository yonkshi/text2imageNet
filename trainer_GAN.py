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


    # Placeholder for the noise that the generator uses to produce a synthesized image
    z = tf.placeholder('float32', [conf.GAN_BATCH_SIZE, 100], name='noise')


    # Encoded texts fed from the pipeline
    datasource = GanDataLoader()
    iterator, next, (label, text_right, real_image) = datasource.correct_pipe()
    iterator_incorrect, next_incorrect, (label2, text_wrong, real_image2) = datasource.incorrect_pipe()
    iterator_txt_G, next_txt_G, (label3, text_G, real_image_G) = datasource.text_only_pipe()


    # Outputs from G and D
    fake_image = generator_resnet(text_G, z)
    S_r = discriminator_resnet(real_image, text_right)
    S_w = discriminator_resnet(real_image2, text_wrong)
    S_f = discriminator_resnet(fake_image, text_G)


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


        encoder_saver = tf.train.import_meta_graph('assets/char-rnn-cnn-19999.meta')
        encoder_saver.restore(sess, 'assets/char-rnn-cnn-19999')


        # Run the initializers for the pipeline
        sess.run([iterator.initializer, iterator_incorrect.initializer, iterator_txt_G.initializer])


        for step in range(epochs):

            # Updating the learning rate every 100 epochs (starting after first 1000 update steps)
            if step != 0 and step > 1000 and (step % decay_every == 0):
                sess.run(tf.assign(lr_v, lr_v * lr_decay))
                log = " ** new learning rate: %f" % (lr * lr_decay)
                print(log)


            # Sample noise, and synthesize image with generator
            z_sample = np.random.normal(0, 1, (conf.GAN_BATCH_SIZE, 100))


            # Updates parameters in G and D, only every third time for D
            print('Update: ', step)
            if step % 3 == 0:
                s_r, s_w, s_f, summary, dloss, gloss, _, _, fake_img_summary = sess.run(
                    [S_r, S_w, S_f, merged, D_loss, G_loss, D_opt, G_opt, fake_img_summary_op],
                    feed_dict={z:z_sample})

                print('Discriminator loss: ', dloss)
                print('Generator loss: ', gloss)

            else:
                s_r, s_w, s_f, summary, gloss, _, fake_img_summary = sess.run(
                    [S_r, S_w, S_f, merged, G_loss, G_opt, fake_img_summary_op], feed_dict={z: z_sample})

                print('Generator loss: ', gloss)


            # Tensorboard stuff
            writer.add_summary(summary, step)

            if step % 10 == 0:
                writer.add_summary(fake_img_summary, step)

            # if step % 1000 == 0 or epoch == epochs-1:
            #     saver.save(sess, 'saved/', global_step=step)


    # Close writer when done training
    writer.close()



# So that we only run things if it's run directly, not if it's imported
if __name__ == '__main__':
    main()
