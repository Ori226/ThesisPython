import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet.neighbours import images2neibs
im_val = np.asarray(np.arange(100.).reshape((1, 1, 10, 10)),dtype=theano.config.floatX)

# Defining variables
images = T.tensor4('images')
neibs = images2neibs(images, neib_shape=(5, 5))

# Constructing theano function 
window_function = theano.function([images], neibs)

# Input tensor (one image 10x10)
im_val = np.asarray(np.arange(100.).reshape((1, 1, 10, 10)),dtype=theano.config.floatX)

# Function application
neibs_val = window_function(im_val)
print ('-----------');
print im_val.shape
print ('-----------');
print neibs_val.shape



from theano.tensor.extra_ops import repeat