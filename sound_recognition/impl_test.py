import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.layers import Input, Embedding, LSTM, Dense, merge, Flatten
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

studypath = './'

unityOutputPath = '../Escape Room/Assets/TensorFlow'
testOutputPath = 'out'

GRAPH_NAME = 'sound_recognition_graph'

X_train = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
y_train = np.array([1, 2, 3])
y_train = to_categorical(y_train)
X_test = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
y_test = np.array([1, 2, 3])
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
    # note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(3,), dtype='int32', name='input_node')
    K.placeholder(shape=(3), name="input_placeholder_x")

    # this embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=512, input_dim=10000, input_length=3)(main_input)

    # a LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(32)(x)

    # we stack a deep fully-connected network on top
    x = Dense(64, activation='relu', name="dense_one")(x) # names are added here
    x = Dense(64, activation='relu', name="dense_two")(x)
    x = Dense(64, activation='relu', name="dense_three")(x)

    x = Flatten()(x)
    # and finally we add the main logistic regression layer
    main_loss = Dense(4, activation='sigmoid', name='output_node')(x)
    model = Model(input=main_input, output=main_loss)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                loss_weights=[1.])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    graph = sess.graph

    # if the node names trigger errors, check these names and use them as parameters of the function
    # print('>>>>>', model.output.op.name)
    # print('<<<<<', model.input.op.name)

    # Replace testOutputPath with unityOutputPath to test in Unity
    # (or move the frozen .bytes file to the unityOutputPath directory)
    export_model(tf.train.Saver(), ["input_node"], "output_node/Sigmoid", sess, unityOutputPath)