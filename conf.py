ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
ALPHA_SIZE = len(ALPHABET)
CHAR_DEPTH = 201
BATCH_SIZE = 40
ENCODER_TRAINING_PATH = 'assets/encoder_train'
PRE_PROCESSING_THREADS = 20

# GAN Config
NUM_D_FILTER = 64
NUM_G_FILTER = 128
ENCODED_TEXT_SIZE = 128
GAN_BATCH_SIZE = 64