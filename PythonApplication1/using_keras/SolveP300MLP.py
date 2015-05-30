
# coding: utf-8

# In[1]:

#read data of a single subject:
import numpy as np 
import scipy.io
import math
import random


# In[2]:

import sys,os; 
sys.path.append('C:\\Users\\ori22_000\\Documents\\IDC-non-sync\\Thesis\\PythonApplication1\\PythonApplication1\\HelperForThesisData\\')
#from HelperForThesisData import LoadThesisData

#FromFileListToArray(all_train_target_files, 'all_target_flatten' ,640)


# In[3]:

import LoadThesisData


# In[4]:

#temp = LoadThesisData.LoadSingleSubject()


# In[5]:

import cPickle as pickle


# In[6]:

#pickle.dump( temp, open( "c:\\temp\\single_subject.p", "wb" ) )


# In[7]:

single_subject_data = pickle.load( open(  "c:\\temp\\single_subject.p", "rb" ) )


# In[8]:

single_subject_data[0]


# In[9]:
from pylab import *
from matplotlib.collections import LineCollection
import mpld3

# In[10]:
target_data_splitted_to_channels =  single_subject_data[0][10].reshape((55,80))
channels_data = [target_data_splitted_to_channels[i,:]+2*i for i in arange(55)]

# In[12]:

non_target_data_splitted_to_channels =  single_subject_data[1][10].reshape((55,80))
channels_data = [non_target_data_splitted_to_channels[i,:]+2*i for i in arange(55)]



# In[14]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils


# In[15]:

all_samples = np.vstack((single_subject_data[0],single_subject_data[1]))
number_of_target_examples =  single_subject_data[0].shape[0]



# In[16]:

number_of_target_examples


# In[17]:

target_tags = np.zeros((number_of_target_examples,1))
non_target_tags = np.ones((number_of_target_examples,1))
all_tags = np.vstack((target_tags,non_target_tags))

# In[19]:
sample_with_tags = np.hstack((all_samples,all_tags))




# In[23]:

#create the model:

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
shuffeled_samples, suffule_tags = shuffle(all_samples,all_tags, random_state=0)

a_train, a_test, b_train, b_test = train_test_split(shuffeled_samples, suffule_tags, test_size=0.66, random_state=42)


# In[24]:
input_dimesion = shuffeled_samples.shape[1]


model = Sequential()
#Input shape: 2D tensor with shape: (nb_samples, input_dim):
# i guess that the nb_samples is the samples dimension
model.add(Dense(input_dimesion, 250, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(250, 50, init='uniform'))
model.add(Dropout(0.3))

model.add(Dense(50, 1, init='uniform'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)


rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
model.compile(loss='mean_absolute_error', optimizer=rmsprop,class_mode='binary')



# In[25]:



model.fit(a_train, b_train, nb_epoch=500,show_accuracy=True)
score = model.evaluate(a_test, b_test)



# In[26]:

[loss, accuracy] = model.test(a_test, b_test, accuracy=True)
print ('loss:', loss)
print ('accuracy:', accuracy)
temp_pauaw = 0
# In[ ]:



