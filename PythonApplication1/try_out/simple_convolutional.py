from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils, generic_utils
from utils import tile_raster_images as show_row_vectors
import keras.datasets.cifar10 
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data(test_split=0.1, seed=113)

number_of_class = 1000
X_train = X_train[0:number_of_class,:,:,:]
y_train = y_train[0:number_of_class,:]


#image size : 32x32

#
nb_classes = 10


model = Sequential()



in_size = 32*32
im_w = 32
im_h = 32
in_filter_w = 3
in_filter_h = 3

model.add(Convolution2D(32, 3, in_filter_h, in_filter_w, border_mode='full')) 
im_w = im_w + in_filter_w - 1
im_h = im_h + in_filter_h -1 



model.add(Activation('relu'))
#model.add(Activation('relu'))


in_filter_w = 5
in_filter_h = 5
model.add(Convolution2D(32, 32, in_filter_h, in_filter_w))

im_w = im_w - in_filter_w + 1
im_h = im_h - in_filter_h + 1 
model.add(Activation('relu'))
#model.add(MaxPooling2D(poolsize=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
#model.add(Activation('relu'))
#model.add(Convolution2D(64, 64, 3, 3)) 
#model.add(Activation('relu'))

pool_w = 1
pool_h = 1

model.add(MaxPooling2D(poolsize=(pool_h, pool_w)))
#model.add(Dropout(0.25))



#filter_count = 3
#input_layer_width = 32
#input_layer_height = 32
#filter_width = 9
#filter_height = 3
#number_of_filters = 32
#pool_width = 2
#pool_height = 2

model.add(Flatten())
#model.add(Dense(200*(80+14)/2, 55))
number_of_params =  32*(im_w*im_h)/(pool_w*pool_h)#   ((input_layer_width+ filter_width-1)*(input_layer_height+ filter_height-1)*number_of_filters)/(pool_width*pool_height)





#model.add(Dense(64*8*8, 256))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

#model.add(Dense(256, 10))
model.add(Dense(number_of_params, 10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

Y_train = np_utils.to_categorical(y_train, nb_classes)
model.fit(X_train, Y_train, nb_epoch=6,batch_size=32)


final_w =  model.layers[2].get_weights()[0]

W_tile_shape=(4,8+1)


import os
import imsave2
img_W_=show_row_vectors(final_w.T,tile_shape=W_tile_shape,tile_spacing=(2,2),img_shape=(int(im_h),int(im_w)+1))
save_dir='MY_AE_tutorial2'
imsave2.imsave2(os.path.join(save_dir,'filters_denoised','4.png'),img_W_)



def get_reconstructed_input(self, hidden):
    """ Computes the reconstructed input given the values of the hidden layer """
    repeated_conv = conv.conv2d(input = hidden, filters = self.W_prime, border_mode='full')

    multiple_conv_out = [repeated_conv.flatten()] * np.prod(self.poolsize)

    stacked_conv_neibs = T.stack(*multiple_conv_out).T

    stretch_unpooling_out = neibs2images(stacked_conv_neibs, self.pl, self.x.shape)

    rectified_linear_activation = lambda x: T.maximum(0.0, x)
    return rectified_linear_activation(stretch_unpooling_out + self.b_prime.dimshuffle('x', 0, 'x', 'x'))