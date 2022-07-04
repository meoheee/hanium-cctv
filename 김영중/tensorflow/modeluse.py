import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('number.h5')
img = cv2.imread('5.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape(28,28,1)
pred =  model.predict(np.array([img]))
print(pd.DataFrame(pred).round(2))
