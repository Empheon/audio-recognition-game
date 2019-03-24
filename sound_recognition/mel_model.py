# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
import librosa
from functions import load_data, split_samples_to_frames

X, XS, y , ys = load_data()

frame_length = 22050
X_split, y_split = split_samples_to_frames(X, y, frame_length)
XS_split, ys_split = split_samples_to_frames(XS, ys, frame_length)

X_split = np.expand_dims(X_split,2)
XS_split = np.expand_dims(XS_split,2)

y_split = to_categorical(y_split)

X_split = np.concatenate((X_split,XS_split[:15]), axis = 0)
y_split = np.concatenate((y_split,ys_split[:15]), axis = 0)

Xmel = []
for idx in range(0, X_split.shape[0]):
    Xmel.append(librosa.feature.melspectrogram(np.ravel(X_split[idx]).astype(float), n_mels = 40)); #128x44
Xmel = np.array(Xmel) 
Xmel = np.expand_dims(Xmel,3)

for a in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(Xmel, y_split, test_size=0.2)
    input_size = Xmel[0].shape

    model = Sequential()
    #model.add(Conv2D(32,(4,4), strides = (32, 32), input_shape = (40,44,1), activation='relu'))
    model.add(Conv2D(8,(3,3), input_shape = (40,44,1), activation='relu'))
    model.add(Conv2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(16,(3,3),activation='relu'))
    model.add(Conv2D(16,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu')) 
    model.add(Dense(2, activation='sigmoid'))
    
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 10, batch_size = 100, validation_data = [X_test, y_test])

    
    
    
    