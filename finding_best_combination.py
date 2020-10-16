import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

result =[]
for i in range(0,1500):

    activ_h = np.random.choice(['elu','softplus'],1)[0] #swish 안됨
    activ_o = np.random.choice(['sigmoid','softsign'],1)[0]
    l2 = 10**(-1*np.random.randint(11))
    a = time.time()
    d1h =np.random.randint(192,300)
    d2h =np.random.randint(96,192)
    d3h =np.random.randint(48,96)
    d4h =np.random.randint(24,48)
    d5h =np.random.randint(6,24)
    d6h =np.random.randint(1,6)
    class AnomalyDetector(Model):
      def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Dense(d1h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d2h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d3h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d4h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d5h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d6h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h)])

        self.decoder = tf.keras.Sequential([
          layers.Dense(d5h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d4h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d3h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d2h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(d1h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
          layers.Dense(301,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_o)])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    autoencoder = AnomalyDetector()
    #opt
    #los = losses.MeanSquaredError()
    autoencoder.compile(optimizer='adam', loss='mae')
    iter_ = np.random.randint(200,1500)
    epo = np.random.randint(4,40)
    bat = np.int(epo*110/iter_)
    if bat == 0:
        bat = 1
    history = autoencoder.fit(train_data, train_data, 
              epochs=epo, 
              batch_size=bat, #Batch사이즈 줄이니까 훅 좋아짐!
              validation_data=(test_data, test_data),
              shuffle=True,
              verbose=0)
    reconstructions = autoencoder.predict(ng_data)
    ng_loss = tf.keras.losses.mae(reconstructions, ng_data)
    threshold = np.min(ng_loss)-0.00000001
    preds = predict(autoencoder, all_data, threshold)
    F_labels = labels_all[preds]
    F_counts = len(labels_all[preds])
    ptime = time.time()-a
    if F_counts <= 8:
        result.append({'Trial': i,'Time': ptime ,'Activation Hidden': activ_h,'Activation Output': activ_o,'L2':l2 ,'NG Loss':ng_loss ,'Epoch':epo ,'Batch Size':bat ,'Fault Counts':F_counts, 'Iteration':epo*110/bat, 'Dimensions': [d1h,d2h,d3h,d4h,d5h,d6h] })
        autoencoder.save_weights('./checkpoints/my_checkpoint/'+str(i)+'th_trial')
res_4 = pd.DataFrame(result)
res_4.to_csv('Result_epochs_batch_0909_3.csv')

#해당 체크포인트 불러오기
#모델 기본정보 res에서 확인
activ_h = 'elu'
activ_o = 'softsign'
l2 = 10**(-8)
a = time.time()
d1h =282
d2h =119
d3h =71
d4h =46
d5h =17
d6h =2
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(d1h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d2h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d3h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d4h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d5h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d6h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h)])

    self.decoder = tf.keras.Sequential([
      layers.Dense(d5h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d4h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d3h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d2h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(d1h,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_h),
      layers.Dense(301,kernel_regularizer=keras.regularizers.l2(l2), activation=activ_o)])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
autoencoder.load_weights('./checkpoints/my_checkpoint/885th_trial')
