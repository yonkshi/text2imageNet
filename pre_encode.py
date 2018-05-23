import tensorflow as tf
from dataloader import *
from lenet.pretrained import generated_lenet
from os import listdir
from scipy.ndimage import imread
from utils import *

def to_img(x):
    ff = np.array(crop_and_flip(imread(x), 500, [500], True), dtype='float32')
    return ff


def main():

    # Load every image in the data set into a list
    images_np = [None]*8189
    rel_path = 'assets/encoder_train/images'
    for i, name in enumerate(listdir(rel_path)):

        if i > 0 and i % 100 == 0:
            print('loaded {} images'.format(i))
            #break

        if name.endswith('.jpg'):
            im_name = rel_path + '/' + name
            index = int(im_name[-9:-4]) - 1
            #im = crop_and_flip(imread(im_name), 500, [500], True)
            images_np[index] = im_name


    pipe = tf.data.Dataset.from_tensor_slices(images_np)
    pipe = pipe.map(lambda x: tf.py_func(to_img, [x], [tf.float32] ), num_parallel_calls=30)
    pipe = pipe.batch(64)

    iter = pipe.make_one_shot_iterator()
    img = iter.get_next()

    # img_out and img_in are placeholders. images_out = sess.run(img_out, {img_in : images})
    encoded_images, image_placeholder = generated_lenet(img) # encoded: N x 1024, img_in: N x 64 x 64 x 3
    encoded_images_out = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with tf.device('/gpu:0'):

            print('starting encoding')
            for i in range(10000):
                try:
                    if i == 2: break
                    zz = sess.run(img[0])
                    encoded_images_out.append(sess.run(encoded_images, feed_dict={image_placeholder:zz}))
                except tf.errors.OutOfRangeError:
                    break
                print(i)
            print('Done!')
        npimg = np.concatenate(encoded_images_out, axis=0)
        np.save('encoded_images', npimg)

if __name__ == '__main__':
    main()