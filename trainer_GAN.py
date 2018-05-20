import datetime
from models import *
from lenet.pretrained import generated_lenet
from dataloader import *
import matplotlib.pyplot as plt
import conf
import tensorflow as tf


def main():
    """The famous main function that no one knows what it's for"""

    # Training parameters
    batch_size = 64
    epochs = 600
    lr = 0.0002
    lr_decay = 0.5
    decay_every = 100
    beta1 = 0.5


    # Parameters for which generator / discriminator to use
    gen_scope = 'generator'
    disc_scope = 'discriminator'
    #gen_scope = 'generator_res'
    #disc_scope = 'discriminator_res'


    # Define placeholders for image and text data
    text_G = tf.placeholder('float32', [batch_size, 1024], name='text_generator')
    text_right = tf.placeholder('float32', [batch_size, 1024], name='encoded_right_text')
    text_wrong = tf.placeholder('float32', [batch_size, 1024], name='encoded_wrong_text')
    real_image = tf.placeholder('float32', [batch_size, 64, 64, 3], name='real_image')
    z = tf.placeholder('float32', [batch_size, 100], name='noise')


    # Outputs from G and D
    fake_image = generator(text_G, z)
    S_r = discriminator(real_image, text_right)
    S_w = discriminator(real_image, text_wrong) # todo: maybe here take real_image2
    S_f = discriminator(fake_image, text_G)


    # Loss functions for G and D
    G_loss = tf.reduce_mean(tf.log(S_f))
    D_loss = tf.reduce_mean(tf.log(S_r) + (tf.log(1 - S_w) + tf.log(1 - S_f))/2)
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


    # to save the graph and all variables
    saver = tf.train.Saver()


    # Write to tensorboard
    merged = tf.summary.merge_all()
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p_GAN")
    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name, tf.get_default_graph())


    # Execute the graph
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        step = 0

        for epoch in range(epochs):

            # Updating the learning rate every 100 epochs (starting after first 1000 update steps)
            if epoch != 0 and step > 1000 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v, lr * new_lr_decay))
                log = " ** new learning rate: %f" % (lr * new_lr_decay)
                print(log)


            # todo: add condition for mini batches
            step += 1


            # Todo: pipeline this shit
            img_r, txt_r, txt_w = 0
            txt_G = 0


            # Sample noise, and synthesize image with generator
            z_sample = np.random.normal(0, 1, (batch_size, 100)) # apparently better to sample like this than with tf
            G_feed = {text_G: txt_G, z: z_sample}
            img_f = sess.run(fake_image, feed_dict=G_feed)


            # Todo: make sure to put the right images. different or same?
            # feed_dicts for the discriminator
            feed_rr = {real_image: img_r, text_right: txt_r}
            feed_rw = {real_image: img_r, text_wrong: txt_w}
            feed_fr = {fake_image: img_f, text_G: txt_G} # the only thing that needs to be feed to get the G_loss

            feed_dict = {**feed_rr, **feed_rw, **feed_fr} # merge these dictionaries


            # Updates parameters in G and D
            dloss, _ = sess.run([D_loss, D_opt], feed_dict=feed_dict)
            gloss, _ = sess.run([G_loss, G_opt], feed_dict=feed_fr)


            # Tensorboard stuff
            summary = sess.run([merged])
            writer.add_summary(summary, step)
            print('Update: ', step)
            print('Discriminator loss: ', dloss)
            print('Generator loss: ', gloss)


            if step % 1000 == 0 or epoch == epochs-1:
                saver.save(sess, './GAN', global_step=step)


    # Close writer when done training
    writer.close()



# So that we only run things if it's run directly, not if it's imported
if __name__ == '__main__':
    main()