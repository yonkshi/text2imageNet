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
import time as ttt
import bisect

from utils import *




def test_gan_pipeline():
    print('hello world')

    l = GanDataLoader()
    encoded, img = l.correct_pipe()
    #iterator2, next2, (encoded2, img2) = l.incorrect_pipe() # , (label2, encoded2, img2)
    #iterator_txt, next_txt, (label3, encoded3, img3) = l.text_only_pipe()

    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p")


    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text_encoder')

    writer = tf.summary.FileWriter('./tensorboard_logs/%s' % run_name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Load pretrained model
        saver = tf.train.import_meta_graph('assets/char-rnn-cnn-19999.meta')
        saver.restore(sess, 'assets/char-rnn-cnn-19999')

        l.preprocess_data_and_initialize(sess)



        print('hey')
        #sess.run(iterator.initializer)
        #sess.run(iterator2.initializer)
        #sess.run(iterator_txt.initializer)
        t0 = time()
        for i in range(100000):


            #_, a, i1 = sess.run([next, encoded, img])
            iter, img_ = sess.run([encoded, img])

           # txt_out = sess.run(encoded3)

            #print(tf.shape(a))
            print(i)
            ttt.sleep(0.1)



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
        test_d = [] # test set only
        train_d = [] # training set only
        t0 = time()
        count = 0
        print('begin processing metadata')
        for class_str in listdir(self.caption_path):
            if 'class' not in class_str: continue # garbage
            c = int(class_str.split('_')[1])-1  # 0 indexed
            text_path = join(self.caption_path, class_str)

            count += 1
            #if count > 5: break # TODO Delete me

            images = []
            for txt_file in listdir(text_path):


                if 'image' not in txt_file: continue  # garbage
                if txt_file.endswith(".txt"):
                    img_name = txt_file.split('.')[0]
                    img_id = int(img_name.split('_')[1])
                    images.append(img_name)

                    # split set
                    if img_id in self.test_set_idx:
                        #test_set_images.append(img_name)
                        test_d.append([class_str, img_name])
                    else:
                        #train_set_images.append(img_name)
                        train_d.append([class_str, img_name])


            d[c] = images

        random.shuffle(test_d)
        random.shuffle(train_d)

        test_idx = np.arange(-len(test_d),0)[:,None]
        train_idx = np.arange(0, len(train_d))[:,None]

        test_d = np.array(test_d)
        train_d = np.array(train_d)

        test_d = np.hstack([test_d, test_idx])
        train_d = np.hstack([train_d, train_idx])

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

    def __init__(self):
        super(GanDataLoader, self).__init__()
        self.processed_images = []
        self.processed_txt = []
        self.trainset_iterators = []
        self.testset_iterators = []
        self._encode_txt(tf.random_normal([10,conf.CHAR_DEPTH, conf.ALPHA_SIZE])) # Prebuild encoder

        self.preprocessed_images_t = tf.placeholder(tf.float32, shape=[None,10, 64, 64, 3], name='preprocessed_images_placeholder') # 4 cropped images of 64x64x3
        self.preprocessed_text_t = tf.placeholder(tf.float32, shape=[None,10, 1024], name='preprocessed_text_placeholder') # 10 captions of 1024 encoded format

    def _load_file(self, label, caption_path, image_path, deterministic=False):
        '''
        File loader for the dataset pipeline

        :param label: data label
        :param caption_path: caption file
        :param image_path: image file
        :return:
        '''

        # Load captions for image
        with open(caption_path, 'r') as txt_file:
            lines = txt_file.readlines()
        line = random.choice(lines)
        if deterministic:
            line = lines[0]
        txt = np.array(self._onehot_encode_text(line),dtype='float32')

        # Load images
        im = imread(image_path, mode='RGB')  # First time for batch
        resized_images = (sample_image_crop_flip(im, deterministic=deterministic) - 127.5)/127.5

        return label, txt, resized_images.astype('float32')


    def preprocess_data_and_initialize(self,sess):
        print('preprocessing training data')
        train_preproc_images, train_preproc_txt = self._preprocess_data(sess, self.trainset_metadata)

        sess.run(self.trainset_iterators, feed_dict={self.preprocessed_images_t: train_preproc_images,
                                                  self.preprocessed_text_t: train_preproc_txt})
        print('preprocessing test data')
        test_preproc_images, test_preproc_txt = self._preprocess_data(sess, self.testset_metadata)
        sess.run(self.testset_iterators, feed_dict={self.preprocessed_images_t: test_preproc_images,
                                                  self.preprocessed_text_t: test_preproc_txt})

    def _preprocess_data(self,  sess, datasource):
        data_size= len(datasource)
        source = tf.data.Dataset.from_tensor_slices(datasource)
        images = source.map(lambda metadata: tf.py_func(self._load_images, [metadata], tf.float32),
                            num_parallel_calls=20)
        images = images.prefetch(100)

        txt = source.map(lambda metadata: tf.py_func(self._load_txt, [metadata],tf.float32),num_parallel_calls=20)
        txt = txt.map(self._encode_txt)
        txt = txt.prefetch(100)

        #
        img_iter = images.make_initializable_iterator()
        image_batches = img_iter.get_next()
        sess.run(img_iter.initializer)

        txt_iter = txt.make_initializable_iterator()
        txt_batches = txt_iter.get_next()
        sess.run(txt_iter.initializer)

        processed_images = []
        processed_txt = []
        print('preprocesing images...')
        c = 0
        while True:
            c += 1
            try:
                img = sess.run(image_batches)
            except tf.errors.OutOfRangeError:
                break

            processed_images.append(img)
            if c % 100 == 0:
                print(c, '/', data_size)

        print('preprocesing text')
        c = 0
        while True:
            c += 1
            try:
                txt = sess.run(txt_batches)
            except tf.errors.OutOfRangeError:
                break
            processed_txt.append(txt)
            if c % 100 == 0:
                print(c, '/', data_size)

        return np.array(processed_images), np.array(processed_txt)

    # New pipeline methods below ------------------
    def _load_images(self, metadata):
        img_file = metadata[1].decode('utf-8') # bytes to string
        image_path = join(self.image_path, img_file + '.jpg')
        im = imread(image_path, mode='RGB')
        images = (sample_image_crop_flip(im, return_multiple=True)- 127.5)/127.5
        return images.astype('float32')
    def _load_txt(self, metadata):
        class_name = metadata[0].decode('utf-8')  # bytes to string
        txt_file = metadata[1].decode('utf-8')  # bytes to string
        caption_path = join(self.caption_path,class_name,txt_file + '.txt')
        with open(caption_path, 'r') as txt_file:
            lines = txt_file.readlines()
        encoded_caps = [self._onehot_encode_text(line) for line in lines]
        txt = np.array(encoded_caps,dtype='float32')
        return txt
    def _encode_txt(self, txt):
        caption_rigid = tf.reshape(txt,[-1,conf.CHAR_DEPTH, conf.ALPHA_SIZE])
        encoded_caption = build_char_cnn_rnn(caption_rigid)
        return encoded_caption

    def _expand_elementwise(self, txt:tf.Tensor):
        txt = tf.expand_dims(txt, 1)
        txt = tf.tile(txt,[1,10,1])
        txt = tf.reshape(txt, [100, 1024])
        return tf.data.Dataset.from_tensor_slices(txt)


    def base_pipe(self, datasource, reuse=False, batch_size = conf.GAN_BATCH_SIZE, deterministic=False, shuffle_txt = False):
        images = tf.data.Dataset.from_tensor_slices(self.preprocessed_images_t)
        images = images.repeat()

        # reusable img pipe
        txt = tf.data.Dataset.from_tensor_slices(self.preprocessed_text_t)
        txt = txt.repeat()

        # Static data ends
        if shuffle_txt:
            txt = txt.shuffle(10000)

        #  === Aligninig texts and images ==
        # tile it 10 times to match dim of image
        # expand 0 dim then flat_map to pipe
        txt = txt.flat_map(self._expand_elementwise)

        # tile 10 times to match dim of txt
        # expand 0 dim then flat_map to pipe
        images = images.flat_map(lambda t: tf.data.Dataset.from_tensor_slices(tf.tile(t,[10,1,1,1])))

        pipe = tf.data.Dataset.zip((txt, images))

        # # If no shuffling before pipeline, pipeline can be cached
        # if not shuffle_txt:
        #     pipe = pipe.cache()

        if not deterministic:
            pipe = pipe.shuffle(10000)

        pipe = pipe.batch(batch_size)
        pipe = pipe.prefetch(200)


        pipe_iter = pipe.make_initializable_iterator()
        pipe_next = pipe_iter.get_next()
        return pipe_iter, pipe_next, pipe
    def correct_pipe(self):
        #correct = tf.data.Dataset.from_generator(self._correct_pair, (tf.int8, tf.string, tf.string))
        correct_iterator, correct_next, _ = self.base_pipe(datasource=self.trainset_metadata, reuse=True)
        (encoded_txt, img) = correct_next
        self.trainset_iterators.append(correct_iterator.initializer)
        return encoded_txt, img
    def incorrect_pipe(self):
        incorrect_iterator, incorrect_next, _ = self.base_pipe(datasource=self.trainset_metadata, reuse=True, shuffle_txt=True)
        (encoded_txt, img) = incorrect_next
        self.trainset_iterators.append(incorrect_iterator.initializer)
        return encoded_txt, img
    def text_only_pipe(self):
        return self.correct_pipe()

    def test_pipe(self, deterministic=False, sample_size=10):
        '''

        :param deterministic: Chooses if the output test is derterministic (defaults to test set 1)
        :return:
        '''
        '''spid out two images'''
        if deterministic:
            ds = self.testset_metadata[0:3]
        else:
            ds = self.testset_metadata
        test_iterator, test_next, pipe = self.base_pipe(datasource=ds,batch_size=sample_size, deterministic=deterministic)
        (encoded_txt, img) = test_next

        self.testset_iterators.append(test_iterator.initializer)
        return encoded_txt, img



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

if __name__ == '__main__':
    test_gan_pipeline()
