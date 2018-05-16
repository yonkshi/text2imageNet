
from models import *
from lenet.pretrained import generated_lenet

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

def main():

    ###======================== DEFIINE MODEL ===================================###
    # t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    # t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    # t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    # t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    # t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    inputseq = ['wtf','wtf']

    txt_encoder = build_char_cnn_rnn()
    lenet_encoded, lenet_image, lenet_model = generated_lenet()



if __name__ == '__main__':
    main()