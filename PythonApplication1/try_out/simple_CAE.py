import theano

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ReverseMaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils, generic_utils
import numpy as np
import random
import pylab
from PIL import Image


import keras.datasets.cifar10 


(X_All, _), (X_test, _) = keras.datasets.cifar10.load_data(test_split=0.1, seed=113)

number_of_samples = 10000
X_train = X_All[0:number_of_samples,:,:,:]
X_test = X_All[number_of_samples:number_of_samples*2,:,:,:]



#image size : 32x32

#
nb_classes = 10


model = Sequential()



in_size = 32*32
im_w = 32
im_h = 32
in_filter_w = 3
in_filter_h = 3





model.add(Convolution2D(10, 3, in_filter_h, in_filter_w, border_mode='full')) 
model.add(Dropout(0.2))
im_w = im_w + in_filter_w - 1
im_h = im_h + in_filter_h -1 
pool_w = 2
pool_h = 2
model.add(Convolution2D(3, 10, in_filter_h, in_filter_w, border_mode='valid')) 
model.add(MaxPooling2D(poolsize=(pool_h, pool_w)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(ReverseMaxPooling2D(poolsize=(pool_h, pool_w))) 
model.add(Flatten())


rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
model.compile(loss='mean_squared_error', optimizer=rmsprop)



#temp = X_train.flatten(1)

flattened_x_trained = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))

model.fit(X_train, flattened_x_trained, nb_epoch=10,batch_size=128)
flattened_x_test = X_train.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))
objective_score = model.evaluate(X_test, flattened_x_test, show_accuracy=True, verbose=1)

reconstruction_results =  model.predict_proba(X_test)
reconstruction_results = reconstruction_results.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],X_test.shape[3])




index1 = random.randint(0,number_of_samples)
index2 = random.randint(0,number_of_samples)
index3 = random.randint(0,number_of_samples)
index4 = random.randint(0,number_of_samples)


pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(X_train[index1,0,:,:]);
pylab.gray();
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(reconstruction_results[index1,0,:,:])



pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(X_train[index2,0,:,:])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(reconstruction_results[index2,0,:,:])


pylab.show()

