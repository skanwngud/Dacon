import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow

import datetime

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, Activation, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.optimizers import Adam

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold


str_time=datetime.datetime.now()

train=np.load('C:/data/dacon/data2/train_set.npy')/255
test=np.load('C:/data/dacon/data2/test_set.npy')/255
ans=pd.read_csv('C:/data/dacon/data2/dirty_mnist_answer.csv', index_col=0)

print(datetime.datetime.now()-str_time)

ans=ans.to_numpy()


# train=train.reshape(-1, 256, 256, 3)

# plt.imshow(train[0])
# plt.imshow(test[0])
# plt.show()

print(train[0])
print(train[0].shape) # (256, 256, 1)
print(ans[0])
print(ans[0].shape)

ans=to_categorical(ans)
ans=ans.reshape(-1, 26, 1, 2)

print(ans[0])
print(ans.shape) # (50000, 26,1, 2)

kf=KFold(n_splits=5, shuffle=True, random_state=23)

# for train_index, val_index in kf.split(train, ans):
#     x_train=train[train_index]
#     x_val=train[val_index]
#     y_train=ans[train_index]
#     y_val=ans[val_index]

x_train, x_val, y_train, y_val=train_test_split(
    train, ans,
    train_size=0.8,
    random_state=23
)

print(datetime.datetime.now()-str_time)

x_train=x_train.reshape(-1, 256, 256, 1)
x_val=x_val.reshape(-1, 256, 256, 1)

print(datetime.datetime.now()-str_time)

model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(256, 256, 1)))
model.add(Flatten())
model.add(Dense(52))
model.add(Reshape((26, 1,2)))
model.add(Dense(2, activation='softmax'))

print(datetime.datetime.now()-str_time)

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['acc']
)

print(datetime.datetime.now()-str_time)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=3
)
# del model

print(datetime.datetime.now()-str_time)

pred=model.predict(test)
print(pred[0])