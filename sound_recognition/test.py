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
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
import librosa
#----------------loading data
X0 = []
XS =[]
y = []
FS = []
clas = 0
max_size = 0
labels = ['clap', 'keys','silence']
for label in labels:
    for idx in range(1,31):
        fs, x = wavfile.read(label + '/' + label + '_' + str(idx) + '.wav')
        if x.size > max_size:
            max_size = x.size
        #stereo
        x = x[:,0]
        if label == 'silence':
            XS.append(x)
        else: 
            X0.append(x)
            FS.append(fs)
            y.append(clas)
    clas = clas + 1
#put zeros
X = []
# for idx in range(0, np.size(X0)):
#     padding = max_size - X0[idx].size
#     X.append(np.pad(X0[idx], (0,padding), 'constant', constant_values=0))
# print(X)
# X = np.array(X).astype(float)
# y = np.array(y)
# FS = np.array(FS)

# X = np.expand_dims(X,2)

# y = to_categorical(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#print(X0)

#Split
# Let's assume fs = 44100 so we have a constant input size
frame_length = 22050
X_split = []
y_split = []
#print(np.size(X0))
for sample_idx in range(0, np.size(X0)):
    sample = X0[sample_idx]
    #print('frame iterations for sample', sample_idx,':',int(len(sample) / frame_length))
    #print('length', len(sample))
    for frame_idx in range(0, int(len(sample) / frame_length) + 1):
        if (len(sample) >= (frame_idx + 1) * frame_length):
            X_split.append(sample[frame_idx * frame_length: (frame_idx + 1) * frame_length])
            y_split.append(y[sample_idx])
        else:
            padding = frame_length - len(sample[frame_idx * frame_length:])
            if padding < int(frame_length / 2): 
                #print(padding)
                X_split.append(np.pad(sample[frame_idx * frame_length:], (0,padding), 'constant', constant_values=0))
                y_split.append(y[sample_idx])
            
#silence
XS_split = []
ys_split = []        
for sample_idx in range(0, np.size(XS)):
    sample_silence = XS[sample_idx]
    #print('frame iterations for sample', sample_idx,':',int(len(sample) / frame_length))
    #print('length', len(sample))
    for frame_idx in range(0, int(len(sample) / frame_length) + 1):
        if (len(sample) >= (frame_idx + 1) * frame_length):
            XS_split.append(sample[frame_idx * frame_length: (frame_idx + 1) * frame_length])
            ys_split.append([0,0])
        else:
            padding = frame_length - len(sample[frame_idx * frame_length:])
            if padding <  int(frame_length / 2):
                XS_split.append(np.pad(sample[frame_idx * frame_length :], (0,padding), 'constant', constant_values=0))
                ys_split.append([0,0])
        
# print(X_split)
X_split = np.array(X_split).astype(float)
y_split = np.array(y_split)
FS = np.array(FS)
X_split = np.expand_dims(X_split,2)
y_split = to_categorical(y_split)
XS_split = np.array(XS_split).astype(float)
XS_split = np.expand_dims(XS_split,2)
ys_split = np.array(ys_split) 

#concatenate silence to other samples
X_split = np.concatenate((X_split,XS_split[:15]), axis = 0)
y_split = np.concatenate((y_split,ys_split[:15]), axis = 0)

#mel
Xmel = []
for idx in range(0, X_split.shape[0]):
    Xmel.append(librosa.feature.melspectrogram(np.ravel(X_split[idx]).astype(float), n_mels = 40)); #128x44
Xmel = np.array(Xmel) 
Xmel = np.expand_dims(Xmel,3)

#MODEL FOR RAW DATA
for a in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2)
    #(26, 44099, 1)
    input_size = X_train[0].shape[0]
    print(X_train.shape)
    #--------------create model
    # one hot encoding
    model = Sequential()
    model.add(Conv1D(32,(32), strides = 32, input_shape = (input_size,1), activation='relu'))
    model.add(Conv1D(32,(8), activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(32,(4),activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu')) 
    model.add(Dense(2, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 10, batch_size = 100, validation_data = [X_test, y_test])

#MODEL FOR MEL
#for a in range(0, 10):
#    X_train, X_test, y_train, y_test = train_test_split(Xmel, y_split, test_size=0.2)
#    #(26, 44099, 1)
#    input_size = Xmel[0].shape
#    print(X_train.shape)
#    #--------------create model
#    # one hot encoding
#    model = Sequential()
#    #model.add(Conv2D(32,(4,4), strides = (32, 32), input_shape = (40,44,1), activation='relu'))
#    model.add(Conv2D(8,(3,3), input_shape = (40,44,1), activation='relu'))
#    model.add(Conv2D(8,(3,3), activation='relu'))
#    model.add(MaxPooling2D((3,3)))
#    model.add(Conv2D(16,(3,3),activation='relu'))
#    model.add(Conv2D(16,(3,3),activation='relu'))
#    model.add(MaxPooling2D((2,2)))
#    model.add(Flatten())
#    model.add(Dense(50, activation='relu')) 
#    model.add(Dense(2, activation='sigmoid'))
#    
#    model.summary()
#    
#    model.compile(loss='binary_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
#    model.fit(X_train, y_train, epochs = 10, batch_size = 100, validation_data = [X_test, y_test])
## print(model.predict(X_test))
# print(y_test)