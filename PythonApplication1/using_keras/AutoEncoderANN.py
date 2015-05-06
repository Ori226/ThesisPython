from __future__ import absolute_import
from __future__ import print_function


import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Optimizer
from keras.utils.theano_utils import shared_zeros, shared_scalar
from keras.utils import np_utils

import theano
import theano.tensor as T

import keras.layers

import keras.objectives
#from keras.optimizers import optimizers
#from keras.objectives import objectives
#import time, copy
#from .utils.generic_utils import Progbar
#from six.moves import range

class MySequntal(Sequential):
    def __init__(self):
        self.layers = []
        self.params = []
        print ('hi')

    def compile(self, optimizer, loss, class_mode="categorical"):
        self.optimizer = keras.optimizers.get(optimizer)
        self.loss = keras.objectives.get(loss)

        self.X = self.layers[0].input # input of model 
        # (first layer must have an "input" attribute!)
        self.y_train = self.layers[-1].output(train=True)
        self.y_test = self.layers[-1].output(train=False)

        # output of model
        self.y = T.matrix() # TODO: support for custom output shapes

        train_loss = self.loss(self.y, self.y_train)
        test_score = self.loss(self.y, self.y_test)

        if class_mode == "categorical":
            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)))
            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)))

        elif class_mode == "binary":
            train_accuracy = T.mean(T.eq(self.y, T.round(self.y_train)))
            test_accuracy = T.mean(T.eq(self.y, T.round(self.y_test)))

        elif class_mode == "regression":        
            train_accuracy  = T.mean(self.y - self.y_train)
            test_accuracy   = T.mean(self.y - self.y_test)
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode

        updates = self.optimizer.get_updates(self.params, train_loss)




        self._train = theano.function([self.X, self.y], train_loss, 
            updates=updates, allow_input_downcast=True,mode=theano.compile.MonitorMode(
                        pre_func=inspect_inputs,
                        post_func=inspect_outputs))
        self._train_with_acc = theano.function([self.X, self.y], [train_loss, train_accuracy], 
            updates=updates, allow_input_downcast=True)
        self._predict = theano.function([self.X], self.y_test, 
            allow_input_downcast=True)
        self._test = theano.function([self.X, self.y], test_score, 
            allow_input_downcast=True)
        self._test_with_acc = theano.function([self.X, self.y], [test_score, test_accuracy], 
            allow_input_downcast=True)
class MyAutoEncoder:
    
    def __init__(self,input_dimesion):
        
        self.model = Sequential()
        model = self.model
        #Input shape: 2D tensor with shape: (nb_samples, input_dim):
        # i guess that the nb_samples is the samples dimension
        model.add(Dense(input_dimesion, input_dimesion, init='uniform'))


        # just a copy paste from the keras tutorial
        layer_s1 = Activation('sigmoid')
        model.add(layer_s1)
        #model.add(Dropout(0.5))
        model.add(Dense(250, input_dimesion, init='uniform'))
        #model.add(Dense(50, 1, init='uniform'))
    
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
        rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
        model.compile(loss='mean_squared_error', optimizer=rmsprop,class_mode='binary')

class MyOptimizer(Optimizer):

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, *args, **kwargs):
        #self.__dict__.update(locals())
        self.iterations = shared_scalar(0)

    def get_updates(self, params, cost):
        grads = self.get_gradients(cost, params)
       # lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        updates = [(self.iterations, self.iterations+1.)]

        for p, g in zip(params, grads):
            #m = shared_zeros(p.get_value().shape) # momentum
            #v = self.momentum * m - lr * g # velocity
            #updates.append((m, v)) 

            #if self.nesterov:
            #new_p = 0.5 # p + self.momentum * v - lr * g
            #else:
            new_p = p
            updates.append((p, new_p))
        return updates







def detect_nan(i, node, fn):
    print ('*** HOWDY ***')


def inspect_inputs(i, node, fn):
    print (i, node, "input(s) value(s):", [input[0] for input in fn.inputs],)

def inspect_outputs(i, node, fn):
    print ("output(s) value(s):", [output[0] for output in fn.outputs])