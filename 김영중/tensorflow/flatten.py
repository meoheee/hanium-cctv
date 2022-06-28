import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt

#예제 데이터 다운로드
(nx,ny),_ = tf.keras.datasets.mnist.load_data()
nx = nx.reshape(60000,784)
ny = pd.get_dummies(ny)
print(nx.shape,ny.shape)
#(imx,imy), _ = tf.keras.datasets.cifar10.load_data()
#imx = imx.reshape(50000,3072)
#imy = pd.get_dummies(imy)
#print(imx.shape,imy.shape)


x = tf.keras.layers.Input(shape=[784])
h = tf.keras.layers.Dense(84, activation='swish')(x)
y = tf.keras.layers.Dense(10, activation='softmax')(h)
model = tf.keras.models.Model(x,y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

model.fit(nx,ny,epochs = 1)
pred = model.predict(nx[0:5])
print(pred)