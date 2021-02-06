import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout

train=pd.read_csv('../data/dacon/mnist/data/mnist_data/train.csv')
test=pd.read_csv('../data/dacon/mnist/data/mnist_data/test.csv')

idx=318
img=train.loc[idx,'0':].values.reshape(28, 28).astype(int)
digit=train.loc[idx, 'digit']
letter=train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()

# train model
x_train=train.drop(['id', 'digit', 'letter'], axis=1).values
x_train=x_train.reshape(-1, 28, 28, 1) # 일반적인 mnist 형태로 변환
x_train=x_train/255 # 전처리

y=train['digit']
y_train=np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit]=1

def create_cnn_model(x_train):
    inputs=Input(x_train.shape[1:])

    bn=BatchNormalization()(inputs)
    conv=Conv2D(128, 5, 1, padding='same', activation='relu')(bn)
    bn=BatchNormalization()(conv)
    conv=Conv2D(128, 2, 1, padding='same', activation='relu')(bn)
    pool=MaxPool2D(2)(conv)

    bn=BatchNormalization()(pool)
    conv=Conv2D(256, 2, 1, padding='same', activation='relu')(bn)
    bn=BatchNormalization()(conv)
    conv=Conv2D(256, 2, 1, padding='same', activation='relu')(bn)
    pool=MaxPool2D(2)(conv)

    flat=Flatten()(pool)

    bn=BatchNormalization()(flat)
    dense=Dense(1000, activation='relu')(bn)

    bn=BatchNormalization()(dense)
    outputs=Dense(10, activation='softmax')(bn)

    model=Model(inputs, outputs)

    return model

model=create_cnn_model(x_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20)

x_test=test.drop(['id', 'letter'], axis=1).values
x_test=x_test.reshape(-1, 28, 28, 1)
x_test=x_test/255

submission=pd.read_csv('../data/dacon/mnist/data/mnist_data/submission.csv')
submission['digit']=np.argmax(model.predict(x_test), axis=1)

submission.to_csv('baseline.csv', index=False)