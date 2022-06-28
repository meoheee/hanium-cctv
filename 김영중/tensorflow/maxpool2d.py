import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

(nx,ny),_ = tf.keras.datasets.mnist.load_data()
nx = nx.reshape(60000,28,28,1)
ny = pd.get_dummies(ny)
print(nx.shape,ny.shape)

x = tf.keras.layers.Input(shape=([28,28,1]))
h = tf.keras.layers.Conv2D(3, kernel_size = 5, activation = 'swish')(x)
h = tf.keras.layers.MaxPool2D()(h)
h = tf.keras.layers.Conv2D(6, kernel_size = 5, activation = 'swish')(h)
h = tf.keras.layers.MaxPool2D()(h)
h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dense(84, activation = 'swish')(h)
y = tf.keras.layers.Dense(10, activation = 'softmax')(h)
model = tf.keras.models.Model(x,y)
model.compile(loss = 'categorical_crossentropy', metrics = 'accuracy')
print(model.summary())

model.fit(nx,ny,epochs = 10)
model.make('number.h5')
img = cv2.imread('0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape(28,28,1)
pred =  model.predict(np.array([img]))
print(pd.DataFrame(pred).round(2))