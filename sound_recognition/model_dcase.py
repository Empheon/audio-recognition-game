# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:27:47 2019

@author: cegra
"""
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import BatchNormalization, Dropout
import os

path = './extracted_data/'
files = os.listdir(path)
mfccs = []
mfccs_labels = []
labels = ['clap', 'keys']
for file in files:
    if labels[0] in file: 
        mfccs_labels.append(0)
    elif labels[1] in file:
        mfccs_labels.append(1)        
    mfccs.append(np.genfromtxt(path + file, delimiter=';'))

mfccs = np.array(mfccs)
mfccs = np.expand_dims(mfccs, axis=3)
mfccs_labels = to_categorical(mfccs_labels)
X_train, X_test, y_train, y_test = train_test_split(mfccs, mfccs_labels, test_size=0.2, random_state=42)

input_shape = (13, 50, 1)
model = Sequential()
model.add(Conv2D(32, (7,7), padding = 'same', data_format = 'channels_last', input_shape = input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((5,5)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (7,7), padding = 'same', data_format = 'channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,3))) #100 was too big for our data 
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(100, activation ='relu',  kernel_initializer = 'uniform'))
model.add(Dropout(0.3))

model.add(Dense(2, activation ='softmax',  kernel_initializer = 'uniform'))

model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs = 200, batch_size = 16, validation_data = [X_test, y_test])
model.summary()