import csv
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


for i in range(0, 20):
  r = np.genfromtxt('extracted_data/keys_'+str(i)+'_features.csv', delimiter=';')

  mfcc = librosa.feature.mfcc(r, sr=16000, n_mfcc=13)
  plt.figure(figsize=(10, 4))
  librosa.display.specshow(mfcc, x_axis='time')
  plt.colorbar()
  plt.title('MFCC')
  plt.tight_layout()
  plt.show()