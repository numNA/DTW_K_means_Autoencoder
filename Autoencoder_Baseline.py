
#Baseline Model 
# https://www.tensorflow.org/tutorials/generative/autoencoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

dataframe = pd.read_csv('NVH_Data.csv')
dataframe = dataframe.drop(['Unnamed: 0'],axis=1)
raw_data = dataframe.values

#3번 클러스터 데이터
dataframe_1 = pd.read_csv('NVH_Data_2.csv')
dataframe_1 = dataframe_1.drop(['Unnamed: 0'], axis=1)
raw_data_1 = dataframe_1.values
raw_data_1.shape
#dataframe_1.tail()
data_1 = raw_data_1[:,:-1]
whole_data_1 = (data_1 - min_val) / (max_val - min_val)
whole_data_1

#나머지 정상 클러스터
dataframe_2 = pd.read_csv('NVH_Data_3.csv')
dataframe_2 = dataframe_2.drop(['Unnamed: 0'], axis=1)
raw_data_2 = dataframe_2.values
raw_data_2.shape
#dataframe_1.tail()
data_2 = raw_data_2[:,:-1]
whole_data_2 = (data_2 - min_val) / (max_val - min_val)
whole_data_2

#데이터 레이블 구분
labels = raw_data[:, -1]
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

#정규화
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

#레이블 정리
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]

anomalous_train_data = train_data[train_labels]
anomalous_test_data = test_data[test_labels]

#모델구축
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(128, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(301, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=40, 
          batch_size=8, #Batch사이즈 줄이니까 훅 좋아짐!
          validation_data=(test_data, test_data),
          shuffle=True)
#학습곡선
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

#Threshold 선정 및 결과확인
threshold = np.mean(train_loss) + np.std(train_loss)*2.445

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.greater(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, preds)))
  print("Precision = {}".format(precision_score(labels, preds)))
  print("Recall = {}".format(recall_score(labels, preds)))
  
preds = predict(autoencoder, whole_data_1, threshold)
