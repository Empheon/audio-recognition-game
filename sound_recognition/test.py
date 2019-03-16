# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:07:52 2019

@author: cegra
"""
from scipy.io import wavfile
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D
from keras.layers.core import Dense, Activation, Flatten

#----------------loading data
X0 = []
y = []
FS = []
clas = 0
max_size = 0
labels = ['clap', 'keys']
for label in labels:
    for idx in range(1,31):
        fs, x = wavfile.read(label + '/' + label + '_' + str(idx) + '.wav')
        if x.size > max_size:
            max_size = x.size
        #stereo
        x = x[:,0]
        FS.append(fs)
        y.append(clas)
        X0.append(x)
    clas = clas + 1
#put zeros
X = []
for idx in range(0, np.size(X0)):
    padding = max_size - X0[idx].size
    X.append(np.pad(X0[idx], (0,padding), 'constant', constant_values=0))

X = np.array(X).astype(float)
y = np.array(y)
FS = np.array(FS)

X = np.expand_dims(X,2)

y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

input_size = X_train[0].shape[0]

#--------------create model
# one hot encoding
model = Sequential()
model.add(Conv1D(32,(256), strides = 256, input_shape = (input_size,1), activation='relu'))
model.add(Conv1D(32,(8), activation='relu'))
model.add(MaxPooling1D(4))
model.add(Conv1D(32,(8),activation='relu'))
model.add(MaxPooling1D(4))
model.add(Flatten())
model.add(Dense(100, activation='relu')) 
model.add(Dense(2, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs = 30, batch_size = 10, validation_data = [X_test, y_test])

print(model.predict(X_test))