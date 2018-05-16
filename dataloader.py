from queue import Queue
from threading import Thread

import numpy as np
from scipy import misc
from scipy.ndimage import imread
import scipy
import conf
from os import listdir
from os.path import isfile, join
import os
from time import time

class DataLoader():
    def __init__(self):

        self.caption_path = join(conf.ENCODER_TRAINING_PATH, 'captions')
        self.image_path = join(conf.ENCODER_TRAINING_PATH, 'images')
        self._load_meta_data()
        self.sh_idx = [] # shuffled index,
        self.data = None


    def _load_meta_data(self):

        d = {}

        for class_str in listdir(self.caption_path):
            if 'class' not in class_str: continue # garbage
            c = int(class_str.split('_')[1])

            text_path = join(self.caption_path, class_str)

            images = []
            for txt_file in listdir(text_path):
                if 'image' not in txt_file: continue  # garbage
                if txt_file.endswith(".txt"):
                    image_name = txt_file.split('.')[0]
                    images.append(image_name)
            d[c] = images

        self.meta_data = d

    def process_data(self):
        t0 = time()
        print('pre processing data')
        # worker thread
        def work(q: Queue, ret_q: Queue):
            while not q.empty():
                cls, img_name = q.get()
                if q.qsize() % 100 == 0: print('remaining', q.qsize())

                img_fpath = join(self.image_path, img_name + '.jpg')
                im = imread(img_fpath, mode='RGB') # First time for batch
                resized_im = resize_image_with_smallest_side(im)

                # Load captions for image
                # TODO what to do with multiple captions per text?
                cls_dir = 'class_%05d' % cls
                txt_fpath = join(self.caption_path, cls_dir, img_name + '.txt')
                with open(txt_fpath, 'r') as txt_file:
                    lines = txt_file.readlines()
                    lines = [l.rstrip() for l in lines]
                txt = list(map(encode_text, lines))

                ret_q.put((cls, resized_im, txt))
                q.task_done()

        threads = []
        data = {}
        in_q = Queue()
        out_q = Queue()

        # Fill worker queue
        for i, (cls, image_names) in enumerate(self.meta_data.items()):
            if i > 41: break # TODO Delete me
            for img_name in image_names:
                in_q.put((cls, img_name))

        # Spawn threads
        for i in range(conf.PRE_PROCESSING_THREADS):
            worker = Thread(target=work, args=(in_q, out_q))
            threads.append(worker)
            worker.start()

        # Blocking for worker threads
        in_q.join()
        while not out_q.empty():
            cls, image, captions = out_q.get()
            if cls not in data:
                data[cls] = []
            data[cls].append((image, captions))

        print('pre processing complete, time:', time() - t0)

        self.data = data


    def _shuffle_idx(self):
        """
        Adds more shuffled index into queue
        :return:
        """
        idx = np.array(list(self.data.keys()))
        np.random.shuffle(idx)
        self.sh_idx += idx.tolist()

    def _encode_text(self, txt):
        l = list(map(c2i, txt))
        l += [0] * (conf.CHAR_DEPTH - len(l))
        return l

    def _c2i(self, c: str):
        return conf.ALPHABET.find(c)

    def next_batch(self):
        batch = []
        if len(self.sh_idx) < conf.BATCH_SIZE:
            self._shuffle_idx()

        for i in range(conf.BATCH_SIZE):
            cls = self.sh_idx.pop()
            d = self.data[cls]
            sample_idx = np.random.randint(0, len(d))
            print(sample_idx)
            img, captions = self.data[cls][sample_idx]


            batch.append((cls, img, captions))

        return batch



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

def load_and_process_captions(): # TODO add batch support
    """
    Loads captions and preprocess them into 1 hot encoding
    :param bathces:
    :return:
    """
    # Test string
    test_strings = ["this is a bird", 'this is another bird'] * 20
    return list(map(encode_text, test_strings)) # maps multiple

def encode_text(txt):
    l = list(map(c2i, txt))
    l += [0] * (conf.CHAR_DEPTH - len(l))
    return l

def c2i(c:str):
    return conf.ALPHABET.find(c)

dd = DataLoader()
dd.process_data()
np.random.seed(100)
for i in range(100):
    b = dd.next_batch()
    print('yo')