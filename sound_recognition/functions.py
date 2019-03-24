# -*- coding: utf-8 -*-
"""
"""
from scipy.io import wavfile
import numpy as np

def load_data():
    """ 
    load stereo samples to an array of mono samples
    XS samples of silence
    X samples if other classes
    y lables
    """
    X = []
    XS =[]
    y = []
    ys = []
    clas = 0
    labels = ['clap', 'keys','silence']
    for label in labels:
        for idx in range(1,31):
            fs, x = wavfile.read(label + '/' + label + '_' + str(idx) + '.wav')
            #stereo
            x = x[:,0]
            if label == 'silence':
                XS.append(x)
                ys.append([0,0])
            else: 
                X.append(x)
                y.append(clas)
        clas = clas + 1
    return X, XS, y, ys
        
        
def split_samples_to_frames(X, y, frame_length):
    """
    split each sample from X into frames of frame_length 
    """
    X_split = []
    y_split = []
    for sample_idx in range(0, np.size(X)):
        sample = X[sample_idx]
        for frame_idx in range(0, int(len(sample) / frame_length) + 1):
            if (len(sample) >= (frame_idx + 1) * frame_length):
                X_split.append(sample[frame_idx * frame_length: (frame_idx + 1) * frame_length])
                y_split.append(y[sample_idx])
            else:
                padding = frame_length - len(sample[frame_idx * frame_length:])
                if padding < int(frame_length / 2): 
                    X_split.append(np.pad(sample[frame_idx * frame_length:], (0,padding), 'constant', constant_values=0))
                    y_split.append(y[sample_idx])
                    
    X_split = np.array(X_split).astype(float)
    y_split = np.array(y_split)
    
    return X_split, y_split
        