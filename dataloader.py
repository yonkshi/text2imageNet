from queue import Queue
from threading import Thread
import random
import datetime

import numpy as np
import tensorflow as tf
from scipy import misc
from scipy.ndimage import imread
from scipy.io import loadmat
from models import build_char_cnn_rnn

import conf
from os import listdir
from os.path import isfile, join
import os
from time import time
import bisect




def test_gan_pipeline():
    print('hello world')
    l = GanDataLoader()
    iterator, next, (label, encoded, img) = l.correct_pipe()
    iterator2, next2, (label2, encoded2, img2) = l.incorrect_pipe()

    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p")


    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text_encoder')

    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Load pretrained model
        saver = tf.train.import_meta_graph('assets/char-rnn-cnn-19999.meta')
        saver.restore(sess, 'assets/char-rnn-cnn-19999')

        sess.run(iterator.initializer)
        sess.run(iterator2.initializer)
        t0 = time()
        for i in range(1000):
            print('run')
            _, a = sess.run([next, encoded])
            _, b = sess.run([next2, encoded2])


            print(tf.shape(a))
            print(tf.shape(b))

        print('time',time()-t0)


class BaseDataLoader:
    def __init__(self):
        self.caption_path = join(conf.ENCODER_TRAINING_PATH, 'captions')
        self.image_path = join(conf.ENCODER_TRAINING_PATH, 'images')
        self.test_set_idx = sorted(loadmat('assets/encoder_train/test_set_idx.mat')['trnid'][0, :])

        self._load_meta_data()
        self.sh_idx = [] # shuffled index,
        self.data = None

    def _load_meta_data(self):
        d = {} # entire set
        test_d = {} # test set only
        train_d = {} # training set only
        t0 = time()
        for class_str in listdir(self.caption_path):
            if 'class' not in class_str: continue # garbage
            c = int(class_str.split('_')[1])-1  # 0 indexed

            text_path = join(self.caption_path, class_str)

            images = []
            test_set_images = []
            train_set_images = []
            for txt_file in listdir(text_path):
                if 'image' not in txt_file: continue  # garbage
                if txt_file.endswith(".txt"):
                    img_name = txt_file.split('.')[0]
                    img_id = int(img_name.split('_')[1])
                    images.append(img_name)

                    # split set
                    if img_id in self.test_set_idx:
                        test_set_images.append(img_name)
                    else:
                        train_set_images.append(img_name)
            d[c] = images

            if test_set_images:
                test_d[c] = test_set_images
            if train_set_images:
                train_d[c] = train_set_images

        self.meta_data = d
        self.testset_metadata = test_d
        self.trainset_metadata = train_d
        print('metadata done:', time() - t0)

    def _onehot_encode_text(self, txt):
        axis1 = conf.ALPHA_SIZE
        axis0 = conf.CHAR_DEPTH
        oh = np.zeros((axis0, axis1))
        for i, c in enumerate(txt):
            if i >= conf.CHAR_DEPTH:
                break # Truncate long text
            char_i = conf.ALPHABET.find(c)
            oh[i, char_i] = 1

        # l = list(map(self._c2i, txt))
        # l += [0] * (conf.CHAR_DEPTH - len(l)) # padding
        return oh

    def _c2i(self, c: str):
        return conf.ALPHABET.find(c)
class GanDataLoader(BaseDataLoader):

    def _incorrect_pair(self):
        '''
        Infinite Generator that generates a sample from incorrect pair of text and image
        :return:
        '''
        random_caption_file = 'haha'
        random_image_file = 'hoho'
        while True:
            # sample caption
            caption_class = random.choice(list(self.trainset_metadata.keys()))
            random_caption_file = random.choice(self.trainset_metadata[caption_class])

            # sample images
            rand_img_cls = random.choice(list(self.trainset_metadata.keys()))
            random_image_file = random.choice(self.trainset_metadata[rand_img_cls])

            # bad condition
            if(random_caption_file == random_image_file): continue
            class_dir = 'class_%05d' % (caption_class + 1) # 1 based index for class dir
            cap_path = os.path.join(self.caption_path,class_dir, random_caption_file + '.txt')
            img_path = os.path.join(self.image_path, random_image_file + '.jpg')
            yield 0, cap_path, img_path

    def _correct_pair(self):
        '''
        Randomly samples a correct pair of images from the pipeline
        :return:
        '''

        while True:
            # sample caption
            random_class = random.choice(list(self.trainset_metadata.keys()))
            random_file = random.choice(self.trainset_metadata[random_class])
            class_dir = 'class_%05d' % (random_class + 1) # 1 based index for class dir
            cap_path = os.path.join(self.caption_path, class_dir, random_file + '.txt')
            img_path = os.path.join(self.image_path, random_file + '.jpg')
            yield 1, cap_path, img_path

    def _load_file(self, label, caption_path, image_path):
        '''
        File loader for the dataset pipeline

        :param label: data label
        :param caption_path: caption file
        :param image_path: image file
        :return:
        '''

        img_file = tf.read_file(image_path)
        cap_file = tf.read_file(caption_path)

        # Load captions for image
        with open(caption_path, 'r') as txt_file:
            lines = txt_file.readlines()
        line = random.choice(lines)
        txt = np.array(self._onehot_encode_text(line),dtype='float32')

        # Load images
        im = imread(image_path, mode='RGB')  # First time for batch
        resized_images = crop_and_flip(im, 64, [80], crop_just_one=True)[0] # TODO Resize


        return label, txt, resized_images

    def _run_encoder(self, label, caption, image):
        caption_rigid = tf.reshape(caption,[-1,conf.CHAR_DEPTH, conf.ALPHA_SIZE])
        encoded_caption = build_char_cnn_rnn(caption_rigid)
        return label, encoded_caption, image

    def base_pipe(self, pipe_in):
        pipe = pipe_in.map(lambda label, txt_file, img_file: tf.py_func(self._load_file, [label, txt_file, img_file],
                                                                   [tf.int8, tf.float32, tf.uint8]),num_parallel_calls=10)
        pipe = pipe.prefetch(100)
        pipe = pipe.batch(conf.GAN_BATCH_SIZE)
        pipe = pipe.map(self._run_encoder)
        pipe = pipe.prefetch(20)

        pipe_iter = pipe.make_initializable_iterator()
        pipe_next = pipe_iter.get_next()
        return pipe_iter, pipe_next
    def correct_pipe(self):

        correct = tf.data.Dataset.from_generator(self._correct_pair, (tf.int8, tf.string, tf.string))
        correct_iterator, correct_next = self.base_pipe(correct)
        (label, encoded_txt, img) = correct_next
        return correct_iterator, correct_next, (label, encoded_txt, img)

    def incorrect_pipe(self):
        incorrect = tf.data.Dataset.from_generator(self._incorrect_pair, (tf.int8, tf.string, tf.string))
        incorrect_iterator, incorrect_next = self.base_pipe(incorrect)
        (label, encoded_txt, img) = incorrect_next
        return incorrect_iterator, incorrect_next, (label, encoded_txt, img)

class DataLoader(BaseDataLoader):
    def __init__(self):
        super(DataLoader, self).__init__()
        self.sh_idx = [] # shuffled index,
        self.data = None


    def process_data(self):
        t0 = time()
        print('pre processing data')
        indices = loadmat('assets/encoder_train/test_set_idx.mat')
        test_set_idx = sorted(loadmat('assets/encoder_train/test_set_idx.mat')['trnid'][0,:])

        # worker thread
        def work(q: Queue, ret_q: Queue):
            while not q.empty():
                cls, img_name = q.get()
                if q.qsize() % 100 == 0: print('remaining', q.qsize())

                # Split test set
                img_id = int(img_name.split('_')[1])
                i = bisect.bisect_left(test_set_idx, img_id)

                is_test_set = False
                if i != len(test_set_idx) and test_set_idx[i] == img_id:
                    is_test_set = True
                    del test_set_idx[i]

                # Load captions for image
                cls_dir = 'class_%05d' % (cls+1)
                txt_fpath = join(self.caption_path, cls_dir, img_name + '.txt')
                with open(txt_fpath, 'r') as txt_file:
                    lines = txt_file.readlines()
                    lines = [l.rstrip() for l in lines]
                txt = list(map(self._onehot_encode_text, lines))

                # Load images
                img_fpath = join(self.image_path, img_name + '.jpg')
                im = imread(img_fpath, mode='RGB') # First time for batch
                resized_images = crop_and_flip(im, crop_just_one=is_test_set)


                if is_test_set and txt:
                    # only 1 image per class
                    ret_q.put((cls, resized_images[0], txt, is_test_set))

                else:
                    for img in resized_images:
                        for caption in txt:
                            ret_q.put((cls, img, caption, is_test_set))
                q.task_done()

        threads = []
        data = {}

        in_q = Queue()
        out_q = Queue()

        # Fill worker queue

        for i, (cls, image_names) in enumerate(self.meta_data.items()):
            for img_name in image_names:
                in_q.put((cls, img_name))

            #if i > 5: break # TODO Delete me

        # Spawn threads
        for i in range(conf.PRE_PROCESSING_THREADS):
            worker = Thread(target=work, args=(in_q, out_q))
            threads.append(worker)
            worker.start()

        # Blocking for worker threads
        in_q.join()
        print('workers completed')
        test_count = 0
        data_count = 0

        test_images = []
        test_labels = []
        test_captions = {}


        while not out_q.empty():
            cls, image, captions, belongs_to_testset = out_q.get()
            if belongs_to_testset:
                #if test_count > 100: continue # TODO Delete me, this is to limit test set for testing purpose

                if cls not in test_captions:
                    test_captions[cls] = []

                test_count += 1
                test_images.append(image)
                test_labels.append(cls)
                test_captions[cls].extend(captions)
            else:
                if cls not in data:
                    data[cls] = []
                data_count += 1
                data[cls].append((image, captions))

        print('pre processing complete, time:', time() - t0)

        self.data = data

        # Convert labels to relative labels
        mapped = list(sorted(test_captions.keys()))
        self.test_labels = list(map(mapped.index,test_labels))
        self.test_images = test_images
        self.test_captions = test_captions

    def _shuffle_idx(self):
        """
        Adds more shuffled index into queue
        :return:
        """
        idx = np.array(list(self.data.keys()))
        np.random.shuffle(idx)
        self.sh_idx += idx.tolist()



    def next_batch(self): #TODO modify to comply with TF pipeline?
        '''
        Get batches of data
        :return:
        '''
        if self.data is None:
            raise Exception('Data not preprocessed! Did you call .process_data() beforehand? ')

        batch = []
        classes = []
        images = []
        captions = []
        if len(self.sh_idx) < conf.BATCH_SIZE:
            self._shuffle_idx()

        for i in range(conf.BATCH_SIZE):
            cls = self.sh_idx.pop()
            d = self.data[cls]
            sample_idx = np.random.randint(0, len(d))
            img, caption = self.data[cls][sample_idx]

            #append
            images.append(img)
            captions.append(caption)
            classes.append(cls)


        return (classes, images, captions)

def load_and_process_image_batch(): # TODO add batch support
    """
    Loads images and preprocess them into 3 channel images
    :param bathces:
    :return: batches of images tensor. [Batchsize, width, height, 3]
    """

    images = []

    im = imread('assets/training/4.png', mode='RGB') # First time for batch
    resized_im = resize_image_with_smallest_side(im)

    images.append(resized_im)

    im = imread('assets/training/4.png', mode='RGB') # First time for batch
    resized_im = resize_image_with_smallest_side(im)
    images.append(resized_im)
    npim = np.array(images)
    return npim

def resize_image_with_smallest_side(image, small_size=224):
    """
    Resize single image array with smallest side = small_size and
    keep the original aspect ratio.

    Author: Qian Ge <geqian1001@gmail.com>

    Args:
        image (np.array): 2-D image of shape
            [height, width] or 3-D image of shape
            [height, width, channels] or 4-D of shape
            [1, height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.
    """
    im_shape = image.shape
    shape_dim = len(im_shape)
    assert shape_dim <= 4 and shape_dim >= 2,\
        'Wrong format of image!Shape is {}'.format(im_shape)

    if shape_dim == 4:
        image = np.squeeze(image, axis=0)
        height = float(im_shape[1])
        width = float(im_shape[2])
    else:
        height = float(im_shape[0])
        width = float(im_shape[1])

    if height <= width:
        new_height = int(small_size)
        new_width = int(new_height/height * width)
    else:
        new_width = int(small_size)
        new_height = int(new_width/width * height)

    if shape_dim == 2:
        im = misc.imresize(image, (new_height, new_width))
    elif shape_dim == 3:
        im = misc.imresize(image, (new_height, new_width, image.shape[2]))
    else:
        im = misc.imresize(image, (new_height, new_width, im_shape[3]))
        im = np.expand_dims(im, axis=0)

    return im

def crop_and_flip(image,os=224, scales = [256], crop_just_one=False):

    """
    :param image: An image on tensor form, h x w x 3
    :param size: output
    :return:
    """

    h, w, c = image.shape



    images = []
    for l in scales:

        im=resize_image_with_smallest_side(image,l)
        h, w, c = im.shape

        if not crop_just_one:
            im_upperleft = im[:os, :os, :]
            images.append(im_upperleft)
            images.append(np.fliplr(im_upperleft))

            im_upperright = im[:os, w-os:, :]
            images.append(im_upperright)
            images.append(np.fliplr(im_upperright))

            im_lowerleft = im[h-os:, :os, :]
            images.append(im_lowerleft)
            images.append(np.fliplr(im_lowerleft))

            im_lowerright = im[h-os:, w-os:, :]
            images.append(im_lowerright)
            images.append(np.fliplr(im_lowerright))

        # crop middle
        im_middle = im[(h - os) // 2:(h + os) // 2, (w - os) // 2:(w + os) // 2, :]
        images.append(im_middle)
        if not crop_just_one:
            images.append(np.fliplr(im_middle))

    #shuffle(images)

    return images


if __name__ == '__main__':
    test_gan_pipeline()