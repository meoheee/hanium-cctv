import tensorflow as tf
import pandas as pd #엑셀등의 파일을 가져올때 쓰는 모듈
import numpy as np
data = pd.read_csv('abc.csv')
data = data.dropna() #빈칸 제거 함수
y = []
x = []
for i, rows in data.iterrows():
    x.append([rows['gre'],rows['gpa'],rows['rank']])
    y.append(rows['admit'])
print(y)
print(x)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation = 'tanh'),#원하는 알고리즘 방법 선택과 노드의 개수 선택
                            tf.keras.layers.Dense(128, activation = 'tanh'),
                            tf.keras.layers.Dense(1, activation = 'sigmoid')]) #결과값은 무조건 확률 하나만 나와야만 하므로 sigmoid 고정

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x),np.array(y),epochs=2000) #x데이터는 인자들, y데이터는 결과값들 넘파이 어레이 혹은 텐서만 가능 epoch는 몇번 학습 시킬거냐
result = model.predict(np.array([[100,2.2,1], [400,4.5,3]]))
print(result)
