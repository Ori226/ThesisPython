import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np





from HelperForThesisData import LoadThesisData

#all_save_samples = LoadThesisData.ReadDataFromMatFiles()
import cPickle as pickle
import time
print 'start'



#pickle.dump( all_save_samples, open( "all_res.p", "wb" ) )


all_save_samples = pickle.load( open( "all_res.p", "rb" ) )


print 'end'






#all_save_samples['target']['train']
#my_non_target_mat = LoadThesisData.ReadDataFromMatFiles('C:\\Users\\ori22_000\\Documents\\Thesis\dataset\\all_falltened_data.mat',['all_non_target_faltten2'])




#my_mat_test = LoadThesisData.ReadDataFromMatFiles('C:\\Users\\ori22_000\\Documents\\Thesis\dataset\\all_falltened_data.mat',['all_target_faltten3'])
#my_non_target_mat_test = LoadThesisData.ReadDataFromMatFiles('C:\\Users\\ori22_000\\Documents\\Thesis\dataset\\all_falltened_data.mat',['all_non_target_faltten3'])

all_target_test = all_save_samples['target']['validation']# my_mat_test['all_target_faltten3']
all_non_target_test = all_save_samples['non_target']['validation']#my_non_target_mat_test['all_non_target_faltten3']


# currently the format is: each raw is a sampe and each column is a featrue
all_target = all_save_samples['target']['train']# my_mat['all_target_faltten2']
all_non_target = all_save_samples['non_target']['train'] #my_non_target_mat['all_non_target_faltten2']
input_dimesion = all_target.shape[1]
number_of_target_examples = all_target.shape[0]
non_target_for_classifcation = all_non_target[0:number_of_target_examples,:]




# just a copy paste from the keras tutorial
all_samples = np.vstack((all_target,non_target_for_classifcation))
target_tags = np.zeros((number_of_target_examples,1))
non_target_tags = np.ones((number_of_target_examples,1))
all_tags = np.vstack((target_tags,non_target_tags))








all_tags_mess = np.vstack((non_target_tags,target_tags))
 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils

model = Sequential()
#Input shape: 2D tensor with shape: (nb_samples, input_dim):
# i guess that the nb_samples is the samples dimension
model.add(Dense(input_dimesion, 250, init='uniform'))


layer_s1 = Activation('sigmoid')
model.add(layer_s1)
model.add(Dropout(0.5))
model.add(Dense(250, 50, init='uniform'))
model.add(Dense(50, 1, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(3, 1, init='uniform'))
#model.add(Activation('softmax'))
    
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)


rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
model.compile(loss='mean_absolute_error', optimizer=rmsprop,class_mode='binary')

# I need to randomally choose subset of the raws and stack them one on the other
#in order to save time I'll start by taking only the first one f=
batch_size = 640*2


# in here, each raw is a sample

X_train =all_samples
y_train = all_tags

temporary_with_tags = np.hstack((X_train,y_train))

temporary_with_tags = np.random.permutation(temporary_with_tags)
X_train = temporary_with_tags[:,:-1]
y_train = temporary_with_tags[:,-1]


model.fit(X_train, y_train, nb_epoch=300, batch_size=batch_size,show_accuracy=True)

print '-------layer--------'
#print model.layers[0].get_weights()




###X_test = X_train
###y_test = all_tags_mess





###number_of_target_examples_test = all_target_test.shape[0]
###non_target_for_classifcation_test = all_non_target_test[0:number_of_target_examples_test,:]

all_target_test = all_save_samples['target']['validation']# my_mat_test['all_target_faltten3']
all_non_target_test = all_save_samples['non_target']['validation']#my_non_target_mat_test['all_non_target_faltten3']
all_samples_test = np.vstack((all_target_test,all_non_target_test))



target_tags_test = np.zeros((all_target_test.shape[0],1))
non_target_tags_test = np.ones((all_non_target_test.shape[0],1))
all_tags_test = np.vstack((target_tags_test, non_target_tags_test))

temporary_with_tags = np.hstack((all_samples_test,all_tags_test))

temporary_with_tags = np.random.permutation(temporary_with_tags)
all_samples_test = temporary_with_tags[:,:-1]
all_tags_test = temporary_with_tags[:,-1]






print('eval---------------')



score = model.evaluate(all_samples_test, all_tags_test, batch_size=all_tags_test.shape[0],show_accuracy=True, verbose=1)

###prediction_results = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
###print(prediction_results)

###import csv

###with open("output.csv", "wb") as f:
###    writer = csv.writer(f,lineterminator='\n')
###    writer.writerows(prediction_results)