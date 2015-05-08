from __future__ import absolute_import, division
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Layer
from keras.optimizers import SGD, RMSprop
import numpy as np


from utils import tile_raster_images as show_row_vectors

import skimage
import sklearn
from sklearn import preprocessing,cross_validation


# -*- coding: utf-8 -*-
import os,sys,pdb
import imsave2
import theano
import theano.tensor as T

from keras import activations, initializations
from keras.utils.theano_utils import shared_zeros, floatX
from keras.utils.generic_utils import make_tuple

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
from scipy import io as sp_io


srng = RandomStreams()



class Dense2(Layer):
    '''
        Just your regular fully connected NN layer.
    '''
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 init='glorot_uniform', 
                 activation='linear', 
                 weights=None,
                 corruption_level=0.3):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim

        
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.hidden_dim))
        self.b = shared_zeros((self.hidden_dim))
        self.b_prime = shared_zeros((self.input_dim))

        numpy_rng = np.random.RandomState(123)

        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.params = [self.W, self.b, self.b_prime]
        self.corruption_level = corruption_level

        if weights is not None:
            self.set_weights(weights)
        



    def output(self, train):
        X = self.get_input(train)

        tilde_x = self.get_corrupted_input(X, self.corruption_level)


        self.W_prime = self.W.T
        encoder_output =   self.activation(T.dot(tilde_x, self.W) + self.b)
        decoder_output = self.activation(T.dot(encoder_output, self.W_prime) + self.b_prime)
        output = decoder_output
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "hidden_dim":self.output_dim,
            "init":self.init.__name__,
            "activation":self.activation.__name__}

    def get_corrupted_input(self, input, corruption_level):
        
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input


from skimage import io


Ols_mat_name='C:\Users\ori22_000\Documents\IDC-non-sync\Thesis\PythonApplication1\PythonApplication1\indian_autoencoder\IMAGES_RAW.mat'
Ols_dict=sp_io.loadmat(Ols_mat_name)
Ols_images=Ols_dict['IMAGESr']



patch_size=(16,16)
Ols_patches=[skimage.util.view_as_windows(im,patch_size,step=4) for im in np.rollaxis(Ols_images,-1,0)]
Ols_patches=np.array(Ols_patches)
n_patches=np.prod(Ols_patches.shape[:3])
nvis=np.prod(Ols_patches.shape[3:])
Ols_patches=Ols_patches.reshape((n_patches,nvis))



random_idx=np.random.permutation(n_patches)
nsamples=40000
kept_idx=random_idx[:nsamples]
kept_patches=Ols_patches[kept_idx,:]


### preprocessing the data
Scaler=preprocessing.StandardScaler(copy=False,with_std=0.33)
processed_patches=Scaler.fit_transform(kept_patches)
processed_patches[processed_patches>1]=1
processed_patches[processed_patches<-1]=-1
processed_patches=(processed_patches+1)*0.4+0.1

## division of train,valid,test sets
train_,valid_test_=cross_validation.train_test_split(processed_patches,
	test_size=0.3,random_state=0)
valid_,test_=cross_validation.train_test_split(valid_test_,test_size=0.5,random_state=0)


##

np.asarray( train_,theano.config.floatX)

train=theano.shared(np.asarray( train_,theano.config.floatX))
test=theano.shared(np.asarray( test_,theano.config.floatX))
valid=theano.shared(np.asarray( valid_,theano.config.floatX),'valid')




X_train = train_

#y_test = np.random.randint(0,2, size=10)

model = Sequential()

nhid = 100


model.add(Dense2(256, nhid, init='glorot_uniform', activation='sigmoid', corruption_level=0.0))
rmsprop = RMSprop(lr=0.01, rho=0.5, epsilon=1e-6)
model.compile(loss='mean_squared_error', optimizer=rmsprop)

model.fit(X_train, X_train, nb_epoch=20, batch_size=500)



final_w =  model.layers[0].get_weights()[0]

W_tile_shape=(int(np.sqrt(nhid)),int(np.sqrt(nhid))+1)
img_W_=show_row_vectors(final_w.T,tile_shape=W_tile_shape,tile_spacing=(2,2),img_shape=patch_size)

save_dir='MY_AE_tutorial'
imsave2.imsave2(os.path.join(save_dir,'filters_denoised','4.png'),img_W_)






