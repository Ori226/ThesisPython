
from skimage import data, io
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy
import numpy as np
import matplotlib.pyplot as plt


import pylab
from PIL import Image

theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature
theano.config.exception_verbosity ='high'

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

def get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        repeated_conv = conv.conv2d(input = hidden, filters = self.W_prime, border_mode='full')

        multiple_conv_out = [repeated_conv.flatten()] * np.prod(self.poolsize)

        stacked_conv_neibs = T.stack(*multiple_conv_out).T

        stretch_unpooling_out = neibs2images(stacked_conv_neibs, self.pl, self.x.shape)

        rectified_linear_activation = lambda x: T.maximum(0.0, x)
        return rectified_linear_activation(stretch_unpooling_out + self.b_prime.dimshuffle('x', 0, 'x', 'x'))

rng = numpy.random.RandomState(23455)

# instantiate 4D tensor for input
input = T.tensor4(name='input')
input.tag.test_value = numpy.random.rand(1,3,639, 516).astype(theano.config.floatX)

# initialize shared variable for weights.
w_shp = (1, 3,3, 3)
w_bound = numpy.sqrt(3 * 3 * 3)
W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name ='W')

temp_w = W.get_value()

#--- create a predefined filter:
#temp_w[0,0,:,:] =  numpy.asarray(np.hstack(( -3*np.ones((3,1)),  8*np.ones((3,1)),  -3*np.ones((3,1)))),dtype=input.dtype)
temp_w[0,0,:,:] =  numpy.asarray([[0,0,0],[0,1,0],[0,0,0]],dtype=input.dtype)
temp_w[0,1,:,:] =  numpy.asarray([[0,0,0],[0,1,0],[0,0,0]],dtype=input.dtype)
temp_w[0,2,:,:] =  numpy.asarray([[0,0,0],[0,1,0],[0,0,0]],dtype=input.dtype)

W.set_value(temp_w)

w_prime_shp = (1, 1,3, 3)
w_prime_bound = numpy.sqrt(1 * 3 * 3)
W_prime = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_prime_shp),
            dtype=input.dtype), name ='W_prime')

temp_w_prime = W_prime.get_value()



#temp_w_prime[0,0,:,:] =  numpy.asarray(np.vstack(( 3*np.ones((1,3)),  -8*np.ones((1,3)),  3*np.ones((1,3)))),dtype=input.dtype)
temp_w_prime[0,0,:,:] =  numpy.asarray([[-4,1,0],[-4,1,0],[0,1,0]],dtype=input.dtype)



W_prime.set_value(temp_w_prime)


b_shp = (1,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv.conv2d(input, W)
f_only_one_conv = theano.function([input],conv_out)


conv_out_prime_t = conv.conv2d(conv_out, W_prime)



#conv_out_prime = conv.conv2d(conv_out_prime_t, W_prime)

f_conv_and_un_conv = theano.function([input],conv_out_prime_t)

#------max pool

#input = T.dtensor4('input')
maxpool_shape = (20, 20)
pool_out = downsample.max_pool_2d(conv_out, maxpool_shape, ignore_border=True)

up_pool_out = pool_out.repeat(20, axis = 2).repeat(20, axis = 3)


#theano.config.exception_verbosity = 'high'


f3 = theano.function([input],up_pool_out)

conv_out_prime2 = conv.conv2d(up_pool_out, W_prime)

conv_out_no_pool_prime2 = conv.conv2d(conv_out, W_prime)

f4 = theano.function([input],conv_out_prime2)
f_rev_conv = theano.function([input],conv_out_no_pool_prime2)







#invals = numpy.random.RandomState(1).rand(3, 2, 5, 5)

#----


##-----
#unpool:        


##----

#def depool(X, factor=2):
#    """ 
#    luke perforated upsample 
#    http://www.brml.org/uploads/tx_sibibtex/281.pdf 
#    """
#    output_shape = [
#        X.shape[1],
#        X.shape[2]*factor,
#        X.shape[3]*factor
#    ]
#    stride = X.shape[2]
#    offset = X.shape[3]
#    in_dim = stride * offset
#    out_dim = in_dim * factor * factor
 
#    upsamp_matrix = T.zeros((in_dim, out_dim))
#    rows = T.arange(in_dim)
#    cols = rows*factor + (rows/stride * factor * offset)
#    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)
 
#    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))
 
#    up_flat = T.dot(flat, upsamp_matrix)
#    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0], output_shape[1], output_shape[2]))
 
#    return upsamp




# open random image of dimensions 639x516


img = Image.open('3wolfmoon.jpg')
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float32') / 256.








# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)

image_1_conv = f_only_one_conv(img_)
image_conv_unconv = f_conv_and_un_conv(img_)
filtered_img = f3(img_)
filtered_img2 = f4(img_)
filtered_img_1_conv = f_only_one_conv(img_)
filtered_img_1_conv_rev = f_rev_conv(img_)

# plot original image and first and second components of output
pylab.subplot(2, 4, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(2, 4, 2); pylab.axis('off'); pylab.imshow(image_1_conv[0, 0, :, :])
pylab.subplot(2, 4, 3); pylab.axis('off'); pylab.imshow(image_conv_unconv[0, 0, :, :])
pylab.subplot(2, 4, 4); pylab.axis('off'); pylab.imshow(filtered_img_1_conv[0, 0, :, :])
pylab.subplot(2, 4, 5); pylab.axis('off'); pylab.imshow(img_[0, 0, :, :])
pylab.subplot(2, 4, 6); pylab.axis('off'); pylab.imshow(filtered_img_1_conv[0, 0, :, :])
pylab.subplot(2, 4, 7); pylab.axis('off'); pylab.imshow(filtered_img_1_conv_rev[0, 0, :, :])
#filtered_img_1_conv
pylab.show()

###io.imshow(filtered_img[0, 0, :, :])
###io.show()



###fig, ax = plt.subplots()
###im = ax.imshow(filtered_img[0, 0, :, :], interpolation='none')
###ax.format_coord = Formatter(im)
###plt.show()
 #def get_reconstructed_input(self, hidden):
 #       """ Computes the reconstructed input given the values of the hidden layer """
 #       repeated_conv = conv.conv2d(input = hidden, filters = self.W_prime, border_mode='full')

 #       multiple_conv_out = [repeated_conv.flatten()] * np.prod(self.poolsize)

 #       stacked_conv_neibs = T.stack(*multiple_conv_out).T

 #       stretch_unpooling_out = neibs2images(stacked_conv_neibs, self.pl, self.x.shape)

 #       rectified_linear_activation = lambda x: T.maximum(0.0, x)
 #       return rectified_linear_activation(stretch_unpooling_out + self.b_prime.dimshuffle('x', 0, 'x', 'x'))