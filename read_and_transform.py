
import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

import pandas as pd
import numpy as np
import librosa
from scipy import signal

seed = 0
numpy.random.seed(seed)
Test_Info = pd.read_csv('Test_Model.csv', index_col='Unnamed: 0')
NG_Info = pd.read_csv('NVH_1800_Test_Model_NG.csv', index_col='Unnamed: 0')

#FFT Transform

n_fft = 600
fs = 16000
i =0
for _ in list_c:
    D_raw = np.array(Test_Info[_])
    D = np.abs(librosa.stft(D_raw[-n_fft:], n_fft=n_fft, hop_length=n_fft+1))
    if i == 0 :
        X_train_1 = D
    else:
        X_train_1=np.hstack((X_train_1,D)) #오늘 가장 중요한 배움!
    i+=1
X_train_1 = X_train_1.transpose()



i =0
for _ in list_ng:
    if i < 4:
        D_raw = np.array(NG_Info[_])
        D = np.abs(librosa.stft(D_raw[-n_fft:], n_fft=n_fft, hop_length=n_fft+1))
        if i == 0 :
            X_ng = D
        else:
            X_ng=np.hstack((X_ng,D))
    else:
        D_raw = np.array(Test_Info[_])
        D = np.abs(librosa.stft(D_raw[-n_fft:], n_fft=n_fft, hop_length=n_fft+1))
        if i == 0 :
            X_ng = D
        else:
            X_ng=np.hstack((X_ng,D))
    i+=1
X_ng = X_ng.transpose()

#Soft-DTW time-seriesk-means Clustering
sdtw_km = TimeSeriesKMeans(n_clusters=8,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred_ng = sdtw_km.predict(X_ng)
fig = plt.figure(figsize=(14, 16))
for yi in range(8):
    plt.subplot(8, 1, yi + 1)
    for xx in XX[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    for xng in X_ng[y_pred_ng == yi]:
        plt.plot(xng.ravel(),"g-")
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 300)
    plt.ylim(0, 120)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)    

plt.tight_layout()
plt.show() 

plt.tight_layout()
plt.show()
