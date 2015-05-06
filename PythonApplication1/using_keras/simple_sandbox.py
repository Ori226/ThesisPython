import  AutoEncoderANN
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
from  theano import config


def sine_wave(input_dimesion =5 ,number_of_samples=3):
    return np.asarray( np.random.rand(number_of_samples,input_dimesion),config.floatX)


model = AutoEncoderANN.MySequntal() 

input_dimesion = 1
number_of_samples = 1
hidden_layer_size = 1
model.add(Dense(input_dimesion, hidden_layer_size, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(hidden_layer_size, input_dimesion, init='uniform'))

    
sgd = AutoEncoderANN.MyOptimizer()# SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
temp = sine_wave(input_dimesion = input_dimesion, number_of_samples = number_of_samples)
X_train = np.asarray( [[1.0]],config.floatX)  #sine_wave(input_dimesion = input_dimesion, number_of_samples = number_of_samples)
rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
model.compile(loss='mean_squared_error', optimizer=sgd,class_mode='regression')
model.fit(X_train, X_train, nb_epoch=2,show_accuracy=True)



#temp  = AutoEncoderANN.MyAutoEncoder()



#print temp.stan


#temp.stan = 6
#print temp.stan


#temp2 = temp

#print temp.stan
#temp2.stan = 7

#print temp.stan



