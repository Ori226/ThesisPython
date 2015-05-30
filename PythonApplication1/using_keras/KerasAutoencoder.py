import numpy as np
import scipy
from scipy import io as sp_io

import skimage
import sklearn
from sklearn import preprocessing,cross_validation

import os,sys,pdb


import theano
from theano import tensor as T


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Layer, initializations, activations
from keras.utils.theano_utils import shared_zeros, floatX
from keras.optimizers import SGD




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





#class Dense2(Layer):
#    '''
#        Just your regular fully connected NN layer.
#    '''
#    def __init__(self, input_dim, hidden_dim, init='glorot_uniform', activation='linear', weights=None):
#        nvis = input_dim
#        nhid = hidden_dim
#        W_shape = nhid,nvis
#        lim=np.sqrt(6./(2*nvis+1))
#        W_init=np.random.uniform(-lim,lim,W_shape)
#        W=theano.shared(W_init)

#        hbias=theano.shared(np.zeros((nhid,1)),broadcastable=[False,True])

#        self.init = initializations.get(init)
#        self.activation = activations.get(activation)
#        self.input_dim = input_dim
        
#        self.hidden_dim = hidden_dim
#        self.output_dim = input_dim

#        self.input = T.matrix()

#        #maybe need to replace the initialization function

#        self.W = self.init((self.input_dim, self.hidden_dim))
#        self.b = shared_zeros((self.hidden_dim))
#        #self.b_tilde = shared_zeros((self.input_dim))

#        self.params = [self.W, self.b]

#        if weights is not None:
#            self.set_weights(weights)

#    def output(self, train):
#        X = self.get_input(train)
#        encoder_output =   self.activation(T.dot(X, self.W) + self.b)
#        decoder_output = self.activation(T.dot(encoder_output, self.W.T) + self.b_tilde)
#        output = decoder_output
#        return output

#    def get_config(self):
#        return {"name":self.__class__.__name__,
#            "input_dim":self.input_dim,
#            "hidden_dim":self.output_dim,
#            "init":self.init.__name__,
#            "activation":self.activation.__name__}








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



## non indiain part:


###

nhid=100




model = Sequential()





model.add(Dense(256, nhid))
model.add(Activation('sigmoid', target=0.05))
model.add(Dense(nhid, 256))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(train_, train_, nb_epoch=3, batch_size=128)

final_w =  model.layers[0].get_weights()[0]

W_tile_shape=(int(np.sqrt(nhid)),int(np.sqrt(nhid))+1)
img_W_=show_row_vectors(final_w.T,tile_shape=W_tile_shape,tile_spacing=(2,2),img_shape=patch_size)

save_dir='MY_AE_tutorial'
imsave2.imsave2(os.path.join(save_dir,'filters','1.png'),img_W_)


