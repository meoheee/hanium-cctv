import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

#예제 데이터 다운로드
(nx,ny),_ = tf.keras.datasets.mnist.load_data()
nx = nx.reshape(60000,28, 28, 1)
ny = pd.get_dummies(ny)
print(nx[0].shape)
#(imx,imy), _ = tf.keras.datasets.cifar10.load_data()
#imx = imx.reshape(50000,3072)
#imy = pd.get_dummies(imy)
#print(imx.shape,imy.shape)


x = tf.keras.layers.Input(shape=[28,28,1])
h = tf.keras.layers.Conv2D(3, 5, activation = 'swish')(x)
h = tf.keras.layers.Conv2D(6, 5, activation = 'swish')(h)
h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dense(84, activation='swish')(h)
y = tf.keras.layers.Dense(10, activation='softmax')(h)
model = tf.keras.models.Model(x,y)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

model.fit(nx,ny,epochs = 50)
model.save('my_model.h5')
img = cv2.imread('0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape(28,28,1)
pred = model.predict(np.array([img]))
print(pd.DataFrame(pred).round(2))