import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense
from keras.layers.core import Permute
from keras.optimizers import SGD
from keras.utils import to_categorical
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# freeze_graph "screenshots" the graph
from tensorflow.python.tools import freeze_graph
# optimize_for_inference lib optimizes this frozen graph
from tensorflow.python.tools import optimize_for_inference_lib

# os and os.path are used to create the output file where we save our frozen graphs
import os
import os.path as path

studypath = './'

unityOutputPath = '../Escape Room/Assets/TensorFlow'
testOutputPath = 'out'

GRAPH_NAME = 'sound_recognition_graph'

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

# Implementation by NemoCpp (Github: https://github.com/mil-tokyo/bc_learning_sound/issues/7)
# Modified for our needs
def EnvNet(input_length=44100, num_classes=2):
    T = 44100 / input_length
    model = Sequential()

    # section 1, conv1
    model.add(
        Conv1D(
            input_shape=(input_length, 1),
            filters=40, 
            kernel_size=8,
            strides=1,
            padding="same",
            name='input_node'
        )
    )

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # section 2, conv2
    model.add(
        Conv1D(
            filters=40, 
            kernel_size=8,
            strides=1,
            padding="same",
        )
    )

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # section 3, max_pooling_2d, 10ms
    # model.add(MaxPooling1D(pool_size=(160)))
    model.add(MaxPooling1D(pool_size=(int(input_length / 100))))
    
    # section 4, swapaxes
    model.add(Permute((2,1)))
    model.add(Reshape((model.output_shape[1], model.output_shape[2], 1)))

    # section 5, conv3
    model.add(
        Convolution2D(
            filters=50, 
            kernel_size=(8,13),
            strides=1,
            data_format="channels_last",
        )
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # section 6, max_pooling_2d
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # section 7, conv4
    model.add(
        Convolution2D(
            filters=50, 
            kernel_size=(1,5),
            strides=1,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # section 8, max_pooling_2d
    # model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(MaxPooling2D(pool_size=(1, int(2 * T))))

    # section 9, F.dropout(F.relu(self.fc5(h)), rain=self.train)
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # section 10, F.dropout(F.relu(self.fc6(h)), rain=self.train)
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # section 11, fc7(h)
    model.add(Dense(num_classes, name='output_node'))

    return model

# -------------------- Begining of script
with tf.Session() as sess:
    X0 = []
    y = []
    # FS = []
    clas = 0
    labels = ['clap', 'keys']
    for label in labels:
        fs, x = wavfile.read('data/' + label + '.wav')

        #stereo
        x = x[:,0]

        max_x = int(max(x))
        min_x = int(min(x))
        n_x = []
        for i in range(0, len(x)):
            n_x.append(2 * ((x[i] - min_x) / (max_x - min_x)) - 1)

        X0.append(n_x)
        # FS.append(fs)
        y.append(clas)
        clas = clas + 1

    print(X0[0][:10])

    X = []


    #Split
    # Let's assume fs = 44100 so we have a constant input size
    frame_length = 44100
    X_split = []
    y_split = []
    print(np.size(X0))
    for sample_idx in range(0, np.size(X0)):
        sample = X0[sample_idx]

        for frame_idx in range(0, int(len(sample) / frame_length)):
            if (len(sample) >= (frame_idx + 1) * frame_length):
                sample_data = sample[frame_idx * frame_length: (frame_idx + 1) * frame_length]

                if (max(sample_data) > 0.2 or min(sample_data) < -0.2):
                    X_split.append(sample_data)
                    y_split.append(y[sample_idx])

    X_split = np.array(X_split).astype(float)
    y_split = np.array(y_split)
    # FS = np.array(FS)
    X_split = np.expand_dims(X_split,2)
    y_split = to_categorical(y_split)

    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2)

    model = EnvNet()

    # print('>>>>>', model.output.op.name)
    # print('<<<<<', model.input.op.name)

    #Paper: step 1: 80, step 2: 20, step 3: 20, step 4: 30 epochs
    print('================== Step 1 ==================')
    sdg = SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer = sdg, metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 1, batch_size = 10, validation_data = [X_test, y_test])

    print('================== Step 2 ==================')
    sdg = SGD(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer = sdg, metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 1, batch_size = 10, validation_data = [X_test, y_test])

    print('================== Step 3 ==================')
    sdg = SGD(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer = sdg, metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 1, batch_size = 10, validation_data = [X_test, y_test])

    print('================== Step 4 ==================')
    sdg = SGD(lr=0.00001)
    model.compile(loss='binary_crossentropy', optimizer = sdg, metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs = 1, batch_size = 10, validation_data = [X_test, y_test])

    export_model(tf.train.Saver(), ["input_node_input"], "output_node/BiasAdd", sess, unityOutputPath)

