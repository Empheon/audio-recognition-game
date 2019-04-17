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

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import os
import os.path as path

studypath = './'

unityOutputPath = '../Escape Room/Assets/TensorFlow'
testOutputPath = 'out'

GRAPH_NAME = 'sound_recognition_graph'

# EXPORT GAPH FOR UNITY
# Function modified from Sara Hanson https://blog.goodaudience.com/tensorflow-unity-how-to-set-up-a-custom-tensorflow-graph-in-unity-d65cc1bd1ab1
def export_model(saver, input_node_names, output_node_name, session, outputPath):
    # creates the 'out' folder where our frozen graphs will be saved
    if not path.exists('out'):
        os.mkdir('out')


    # GRAPH SAVING - '.pbtxt'
    tf.train.write_graph(session.graph_def, 'out', GRAPH_NAME + '_graph.pbtxt')

    # GRAPH SAVING - '.chkp'
    # KEY: This method saves the graph at it's last checkpoint (hence '.chkp')
    saver.save(session, 'out/' + GRAPH_NAME + '.chkp')
    

    # GRAPH SAVING - '.bytes'
    freeze_graph.freeze_graph('out/' + GRAPH_NAME + '_graph.pbtxt', None, False,
                              'out/' + GRAPH_NAME + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              outputPath + '/frozen_' + GRAPH_NAME + '.bytes', True, "")

    print("graph saved!")


with tf.Session() as sess:
    path = './recorded_data/'
    files = os.listdir(path)
    mfccs = []
    mfccs_labels = []
    bg_labels = []
    labels = ['clap', 'keys', 'bag']
    for file in files:
        if labels[0] in file: 
            mfccs_labels.append(0)
        elif labels[1] in file:
            mfccs_labels.append(1)  
        elif labels[2] in file:
            mfccs_labels.append(2)
        if labels[0] in file or labels[1] in file or labels[2] in file:
            mfccs.append(np.genfromtxt(path + file, delimiter=';'))

    mfccs = np.array(mfccs)
    mfccs = np.expand_dims(mfccs, axis=3)
    mfccs_labels = to_categorical(mfccs_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(mfccs, mfccs_labels, test_size=0.05, random_state=42)

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', data_format = 'channels_last', input_shape = (12, 50, 1), name='input_node'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3,10)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3), padding = 'same', data_format = 'channels_last'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((4,5)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(100, activation ='relu',  kernel_initializer = 'uniform'))
    model.add(Dropout(0.3))

    model.add(Dense(3, activation ='sigmoid',  kernel_initializer = 'uniform', name='output_node'))

    # Use to read the real layer names
    # print('>>>>>', model.output.op.name)
    # print('<<<<<', model.input.op.name)
    
    model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 30, batch_size = 16, validation_data = [X_test, y_test])
    # model.summary()

    export_model(tf.train.Saver(), ["input_node_input"], "output_node/Sigmoid", sess, unityOutputPath)