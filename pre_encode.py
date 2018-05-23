import tensorflow as tf
from dataloader import *
from lenet.pretrained import generated_lenet
from os import listdir
from scipy.ndimage import imread
from utils import *


def main():

    # Load every image in the data set into a list
    images_np = np.zeros((8188, 500, 500, 3))
    rel_path = 'assets/encoder_train/images'
    for i, name in enumerate(listdir(rel_path)):

        if i > 0 and i % 100 == 0:
            print('loaded {} images'.format(i))
            #break

        if name.endswith('.jpg'):
            im_name = rel_path + '/' + name
            index = int(im_name[-9:-4]) - 1
            im = crop_and_flip(imread(im_name), 500, [500], True)
            images_np[index] = im



    # img_out and img_in are placeholders. images_out = sess.run(img_out, {img_in : images})
    encoded_images, img_in, _ = generated_lenet() # encoded: N x 1024, img_in: N x 64 x 64 x 3

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        with tf.device('/gpu:0'):

            print('starting encoding')
            encoded_images_out = sess.run(encoded_images, feed_dict={img_in: images_np})
            print('Done!')

        np.save('encoded_images', encoded_images_out)

if __name__ == '__main__':
    main()