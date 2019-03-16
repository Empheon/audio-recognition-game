import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import SGD
#from keras.layers import Input, Embedding, LSTM, Dense, merge, Flatten
from keras.models import Model
from keras.utils import to_categorical

from keras import backend as K

# freeze_graph "screenshots" the graph
from tensorflow.python.tools import freeze_graph
# optimize_for_inference lib optimizes this frozen graph
from tensorflow.python.tools import optimize_for_inference_lib

# os and os.path are used to create the output file where we save our frozen graphs
import os
import os.path as path

import numpy as np

from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D
from keras.layers.core import Dense, Activation, Flatten
import librosa

studypath = './'

unityOutputPath = '../Escape Room/Assets/TensorFlow'
testOutputPath = 'out'

GRAPH_NAME = 'sound_recognition_graph'

X_train = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [15, 40, 2, 0]])
y_train = np.array([1, 2, 3])
# y_train = to_categorical(y_train)
X_test = np.array([[1, 2, 3, 4], [15, 40, 2, 0], [4, 5, 6, 7]])
y_test = np.array([1, 3, 2])
y_test = to_categorical(y_test)


# EXPORT GAPH FOR UNITY
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

    # GRAPH OPTIMIZING
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(outputPath + '/frozen_' + GRAPH_NAME + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + GRAPH_NAME + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")





with tf.Session() as sess:
    # for i in range(0, 10):
    #     X_train = np.append(X_train, X_train, axis=0)
    #     y_train = np.append(y_train, y_train, axis=0)
    
    # print(y_train.shape)
    # y_train = to_categorical(y_train)
    # print(X_train.shape)
    # print(y_train.shape)

    # print(X_test.shape)
    # print(y_test.shape)
    # # note that we can name any layer by passing it a "name" argument.
    # main_input = Input(shape=(4,), dtype='int32', name='input_node')
    # K.placeholder(shape=(4), name="input_placeholder_x")

    # # this embedding layer will encode the input sequence
    # # into a sequence of dense 512-dimensional vectors.
    # x = Embedding(output_dim=512, input_dim=10000, input_length=4)(main_input)

    # # a LSTM will transform the vector sequence into a single vector,
    # # containing information about the entire sequence
    # lstm_out = LSTM(32)(x)

    # # we stack a deep fully-connected network on top
    # x = Dense(64, activation='relu', name="dense_one")(x) # names are added here
    # x = Dense(64, activation='relu', name="dense_two")(x)
    # x = Dense(64, activation='relu', name="dense_three")(x)

    # x = Flatten()(x)
    # # and finally we add the main logistic regression layer
    # main_loss = Dense(4, activation='sigmoid', name='output_node')(x)
    # model = Model(input=main_input, output=main_loss)
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy',
    #             loss_weights=[1.])

    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)






    #----------------loading data
    # X0 = []
    # y = []
    # FS = []
    # clas = 0
    # max_size = 0
    # labels = ['clap', 'keys']
    # for label in labels:
    #     for idx in range(1,11):
    #         fs, x = wavfile.read(label + '/' + label + '_0' + str(idx) + '.wav')
    #         if x.size > max_size:
    #             max_size = x.size
    #         #stereo
    #         x = x[:,0]
    #         FS.append(fs)
    #         y.append(clas)
    #         X0.append(x)
    #     clas = clas + 1
    # #put zeros
    # X = []
    # for idx in range(0, np.size(X0)):
    #     padding = max_size - X0[idx].size
    #     X.append(np.pad(X0[idx], (0,padding), 'constant', constant_values=0))

    # X = np.array(X).astype(float)
    # y = np.array(y)
    # FS = np.array(FS)

    # X = np.expand_dims(X,2)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # input_size = X_train[0].shape[0]

    # #--------------create model
    # model = Sequential()
    # model.add(Conv1D(32,(256), strides = 256, input_shape = (input_size,1), activation='relu', name='input_nodee'))
    # model.add(Conv1D(32,(8), activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Conv1D(32,(8),activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu')) 
    # model.add(Dense(1, activation='sigmoid', name='output_node'))

    # model.summary()

    # model.compile(loss='binary_crossentropy', optimizer = 'SGD', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs = 20, batch_size = 10, validation_data = [X_test, y_test])

    # #print(model.predict(X_train[0]))



    # tf.placeholder(tf.float32, shape=[input_size,1], name="input_placeholder_x")





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
    frame_length = 1024
    X_split = []
    y_split = []
    print(np.size(X0))
    for sample_idx in range(0, np.size(X0)):
        sample = X0[sample_idx]
        print('frame iterations for sample', sample_idx,':',int(len(sample) / frame_length))
        print('length', len(sample))
        for frame_idx in range(0, int(len(sample) / frame_length)):
            if (len(sample) >= (frame_idx + 1) * frame_length):
                X_split.append(sample[frame_idx * frame_length: (frame_idx + 1) * frame_length])
                y_split.append(y[sample_idx])

    # print(X_split)

    X_split = np.array(X_split).astype(float)
    y_split = np.array(y_split)
    FS = np.array(FS)
    X_split = np.expand_dims(X_split,2)
    y_split = to_categorical(y_split)

    # for a in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2)
    #(26, 44099, 1)
    input_size = X_train[0].shape[0]
    print(X_train.shape)
    #--------------create model
    # one hot encoding
    model = Sequential()
    model.add(Conv1D(32,(32), strides = 32, input_shape = (input_size,1), activation='relu', name='input_node'))
    model.add(Conv1D(32,(8), activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(32,(4),activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu')) 
    model.add(Dense(2, activation='sigmoid', name='output_node'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 10, batch_size = 100, validation_data = [X_test, y_test])



    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # if the node names trigger errors, check these names and use them as parameters of the function
    print('>>>>>', model.output.op.name)
    print('<<<<<', model.input.op.name)

    # Replace testOutputPath with unityOutputPath to test in Unity
    # (or move the frozen .bytes file to the unityOutputPath directory)
    # testOutputPath
    export_model(tf.train.Saver(), ["input_node_input"], "output_node/Sigmoid", sess, unityOutputPath)