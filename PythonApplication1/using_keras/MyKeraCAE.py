#import sys
#import os
#sys.path.append(os.path.split(sys.argv[0])[0])
theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature
theano.config.exception_verbosity ='high'


import ReverseMaxDepool

import MyUtils


import keras.datasets.cifar10 
import numpy as np
import scipy
from scipy import io as sp_io

import skimage
import sklearn
from sklearn import preprocessing,cross_validation

import os,sys,pdb

from PIL import Image


import theano
from theano import tensor as T




from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Layer, initializations, activations
from keras.utils.theano_utils import shared_zeros, floatX
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import pylab
import matplotlib.pyplot as plt


class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z[0])







parentdir = os.path.dirname(__file__)
sys.path.insert(0,parentdir)


#from indian_autoencoder.custom_io import custom_io

#from custom_io.utils import tile_raster_images as show_row_vectors
from utils import tile_raster_images as show_row_vectors
#from custom_io.imsave2 import imsave2
import imsave2
#from custom_io.imsave2 import savefig2
from imsave2 import savefig2

#from custom_io.utils import plot_together
from utils import plot_together

from collections import OrderedDict






#read the fucking wolf image:

img = Image.open('3wolfmoon.jpg')
# dimensions are (height, width, channel)
img = np.asarray(img, dtype='float32') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)

#pylab.imshow(img_[0, 0, :, :])

fig, ax = plt.subplots()
im = ax.imshow(img, interpolation='none')
ax.format_coord = Formatter(im)
plt.show()



in_size = 32*32
#im_w = 32
#im_h = 32
#in_filter_w = 3
#in_filter_h = 3

#model.add(Convolution2D(32, 3, in_filter_h, in_filter_w, border_mode='full')) 
#im_w = im_w + in_filter_w - 1
#im_h = im_h + in_filter_h -1 

result_image = MyUtils.ImageToTheanoTensor('3wolfmoon.jpg')
two_samples = np.vstack((result_image,result_image))

nhid=100
model = Sequential()
model.add(Convolution2D(1, 2, 3, 3, border_mode='full'))
#model.add(Convolution2D(1, 1, 3, 3, border_mode='valid'))
#model.add(Convolution2D(3, 1, 3, 3, border_mode='full'))
model.add(Activation('sigmoid', target=0.05))
#model.add(Dense(nhid, 256))


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data(test_split=0.1, seed=113)


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, X_train, nb_epoch=3, batch_size=128)

final_w =  model.layers[0].get_weights()[0]

W_tile_shape=(int(np.sqrt(nhid)),int(np.sqrt(nhid))+1)
img_W_=show_row_vectors(final_w.T,tile_shape=W_tile_shape,tile_spacing=(2,2),img_shape=patch_size)

save_dir='MY_AE_tutorial'
imsave2.imsave2(os.path.join(save_dir,'filters','1.png'),img_W_)


