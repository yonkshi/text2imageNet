import tensorflow as tf
from dataloader import *
from lenet.pretrained import generated_lenet
from os import listdir
from scipy.ndimage import imread
from utils import *


def onehot_encode_text(txt):
    axis1 = conf.ALPHA_SIZE
    axis0 = conf.CHAR_DEPTH
    oh = np.zeros((axis0, axis1))
    for i, c in enumerate(txt):
        if i >= conf.CHAR_DEPTH:
            break  # Truncate long text
        char_i = conf.ALPHABET.find(c)
        oh[i, char_i] = 1
    return oh



### Take for example k=5 texts, encode them with cnnrnn
### feed all images n=8159 (or something like this) to googlenet.
### pick resulting best mathcing pictures. With highest compatibility scores. Retrieval!

def main():

    # Test cases
    text_names = ['assets/encoder_train/captions/class_00001/image_06734.txt',
                  'assets/encoder_train/captions/class_00009/image_06396.txt',
                  'assets/encoder_train/captions/class_00017/image_03830.txt',
                  'assets/encoder_train/captions/class_00034/image_06929.txt',
                  'assets/encoder_train/captions/class_00054/image_05399.txt']

    texts = np.array([onehot_encode_text(txt) for txt in text_names], dtype=np.float32)
    texts = tf.convert_to_tensor(texts)

    encoded_texts = build_char_cnn_rnn(texts) # k x 1024


    # Load every image in the data set into a list
    image_list = []
    rel_path = 'assets/encoder_train/images'
    for i, name in enumerate(listdir(rel_path)):

        if i > 0 and i % 100 == 0:
            print('loaded {} images'.format(i))
            break

        if name.endswith('.jpg'):
            im_name = rel_path + '/' + name
            im = crop_and_flip(imread(im_name), 500, [500], True)
            image_list.append(im)


    images_np = np.array(image_list)


    # img_out and img_in are placeholders. images_out = sess.run(img_out, {img_in : images})
    encoded_images, img_in, _ = generated_lenet() # encoded: N x 1024, img_in: N x 64 x 64 x 3

    # Calculate scores. N images and k texts.
    scores = tf.matmul(encoded_images, tf.matrix_transpose(encoded_texts)) # N x k

    # Indices for the predicted matching images
    predictions = tf.argmax(scores, axis=0)




    graph = tf.get_default_graph()
    variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='txt_encode')

    # Only fill up these variables if using saver.restore. GOOD!
    saver = tf.train.Saver(variables)


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'assets/char-rnn-cnn-19999')
        print('variables restored')

        encoded_texts, encoded_images_out, predictions, scores = sess.run([encoded_texts, encoded_images, predictions, scores],
                                                                  feed_dict={img_in: images_np})

        best_images = images_np[predictions]
        a = 0

        writer = tf.summary.FileWriter(logdir='graphs', graph=sess.graph)


if __name__ == '__main__':
    main()