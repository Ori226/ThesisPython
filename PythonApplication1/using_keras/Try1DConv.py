import scipy.io
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD



mat_no_target = scipy.io.loadmat('C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\non_target_RSVP_Color116msVPfat.mat')
mat = scipy.io.loadmat('C:\\Users\\ori22_000\\Documents\\Thesis\\code_and_matlab\\flattened_data\\all_target_RSVP_Color116msVPgcf.mat')
all_target_metrix = mat['all_target_flatten']




all_non_target_metrix = mat_no_target['all_non_target_flatten']
np.random.shuffle(all_non_target_metrix)
all_non_target_metrix = all_non_target_metrix[0:all_target_metrix.shape[0],:]




all_target_metrix = mat['all_target_flatten']
number_of_target_examples = all_target_metrix.shape[0]
number_of_non_target_examples = all_non_target_metrix.shape[0]

target_tags = np.zeros((number_of_target_examples,1))
non_target_tags = np.ones((number_of_non_target_examples,1))
all_tags = np.vstack((target_tags,non_target_tags))

Y_train = all_tags
all_samples =  np.vstack((all_target_metrix,all_non_target_metrix))


print(all_target_metrix.shape)

#currently not choosed randomally
#x_row = all_target_metrix[0:50,:]
x  = all_samples.reshape(all_samples.shape[0],55,1,80)

model = Sequential()
model.add(Convolution2D(200, 55, 1, 15, border_mode='full')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(1, 2)))
#200x1x40
model.add(Flatten())
model.add(Dense(200*(80+14)/2, 55))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(55, 1))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)



model.fit(x, Y_train, batch_size=32, nb_epoch=100,  show_accuracy=True)




