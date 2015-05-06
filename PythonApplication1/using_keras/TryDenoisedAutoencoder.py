from __future__ import absolute_import, division
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Layer
from keras.optimizers import SGD
import numpy as np




import skimage
import sklearn
from sklearn import preprocessing,cross_validation


# -*- coding: utf-8 -*-


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
    def __init__(self, input_dim, hidden_dim, init='glorot_uniform', activation='linear', weights=None):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim

        
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.hidden_dim))
        self.b = shared_zeros((self.hidden_dim))
        self.b_tilde = shared_zeros((self.input_dim))

        self.params = [self.W, self.b, self.b_tilde]

        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        encoder_output =   self.activation(T.dot(X, self.W) + self.b)
        decoder_output = self.activation(T.dot(encoder_output, self.W.T) + self.b_tilde)
        output = decoder_output
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "hidden_dim":self.output_dim,
            "init":self.init.__name__,
            "activation":self.activation.__name__}


from skimage import io
def GenerateToyExampleData():    
    top_image = np.vstack(( np.ones((5,10)),np.zeros((5,10))))
    bottom_image = np.vstack((np.zeros((5,10)), np.ones((5,10))))
    left_image = np.hstack((np.ones((5,10)), np.zeros((5,10))))
    return [top_image,bottom_image,left_image]

all_image = GenerateToyExampleData()

flattened = [im.flatten() for im in all_image]


Ols_mat_name='C:\Users\ori22_000\Documents\IDC-non-sync\Thesis\PythonApplication1\PythonApplication1\indian_autoencoder\IMAGES_RAW.mat'
Ols_dict=sp_io.loadmat(Ols_mat_name)
Ols_images=Ols_dict['IMAGESr']






patch_size=(16,16)
Ols_patches=[skimage.util.view_as_windows(im,patch_size,step=4) for im in np.rollaxis(Ols_images,-1,0)]
Ols_patches=np.array(Ols_patches)
n_patches=np.prod(Ols_patches.shape[:3])
nvis=np.prod(Ols_patches.shape[3:])
Ols_patches=Ols_patches.reshape((n_patches,nvis))






X_train = np.random.rand(10,64)

y_test = np.random.randint(0,2, size=10)

model = Sequential()





model.add(Dense2(64, 64, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense2(64, 64, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense2(64, 1, init='uniform'))
#model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, X_train, nb_epoch=20, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)






