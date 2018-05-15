#from googlenet.googlenet_custom_layers import PoolHelper, LRN
from keras.models import model_from_json
from scipy.misc import imread
import numpy as np

# model = model_from_json(open('googlenet/googlenet_architecture.json').read(), custom_objects={"PoolHelper": PoolHelper, "LRN": LRN})
# model.load_weights('googlenet/googlenet_weights.h5')

img = imread('kitten.png', mode='RGB')
height,width = img.shape[:2]
img = img.astype('float32')
# subtract means
img[:, :, 0] -= 123.68
img[:, :, 1] -= 116.779
img[:, :, 2] -= 103.939
img[:,:,[0,1,2]] = img[:, :, [2, 1, 0]] # swap channels
img = img.transpose((2, 0, 1)) # re-order dimensions
img = img[:,(height-224)//2:(height+224)//2, (width-224)//2:(width+224)//2] #crop
img = np.expand_dims(img, axis=0) # add dimension for batch

print(img)