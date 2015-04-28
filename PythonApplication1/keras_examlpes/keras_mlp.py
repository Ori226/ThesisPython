import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop


model = Sequential()
model.add(Dense(2, 1, init='uniform'))
model.add(Activation('sigmoid'))
#model.add(Dense(3, 1, init='uniform'))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(3, 3, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(3, 1, init='uniform'))
#model.add(Activation('softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.09, nesterov=True)


rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
model.compile(loss='mean_absolute_error', optimizer=rmsprop,class_mode='binary')

X_train = np.array([[1,1],[-1,-1],[1,-1],[1,-1]])
y_train = np.array([[0],[0],[1],[1]])

model.fit(X_train, y_train, nb_epoch=300, show_accuracy=True)
#score = model.evaluate(X_test, y_test, batch_size=1)

print(model.predict_classes(X_train))