
# coding: utf-8

import numpy as np 
import scipy.io
import math
import random

from pylab import *
from matplotlib.collections import LineCollection



import sys,os; 
# I should learn how to load libraries in a more elegant way
sys.path.append('C:\\Users\\ori22_000\\Documents\\IDC-non-sync\\Thesis\\PythonApplication1\\PythonApplication1\\HelperForThesisData\\')

import LoadThesisData
import cPickle as pickle

'''
load the data from the mat file and save it to a "Pickle" file, or,
load it if it already pickled
'''
load_data_from_mat_file = False
if (load_data_from_mat_file):
    single_subject_data = LoadThesisData.LoadSingleSubject()
    pickle.dump( single_subject_data, open( "c:\\temp\\single_subject.p", "wb" ) )
else:
    single_subject_data = pickle.load( open(  "c:\\temp\\single_subject.p", "rb" ) )

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils

all_samples = np.vstack((single_subject_data[0],single_subject_data[1]))
number_of_target_examples =  single_subject_data[0].shape[0]

'''
Create the tagging column
'''
target_tags = np.zeros((number_of_target_examples,1))
non_target_tags = np.ones((number_of_target_examples,1))
all_tags = np.vstack((target_tags,non_target_tags))

'''
just to validate the shape:
'''
all_tags.shape

sample_with_tags = np.hstack((all_samples,all_tags))


'''
suffle the samples in order to balance between the target and non target:
'''
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
shuffeled_samples, suffule_tags = shuffle(all_samples,all_tags, random_state=0)
'''
split the data to train and validation
'''
a_train, a_test, b_train, b_test = train_test_split(shuffeled_samples, suffule_tags, test_size=0.66, random_state=42)

'''
define the neural network model:
'''

input_dimesion = shuffeled_samples.shape[1]


model = Sequential()

model.add(Dense(input_dimesion, 250, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(250, 50, init='uniform'))
model.add(Dense(50, 1, init='uniform'))

'''
define the optimization function and compile it:
'''

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
model.compile(loss='mean_absolute_error', optimizer=rmsprop,class_mode='binary')

'''
train the mocdel
'''
print model.fit(a_train, b_train, nb_epoch=500,show_accuracy=True,verbose=1)


'''
test the model
'''
print model.test(a_test, b_test, accuracy=True)






