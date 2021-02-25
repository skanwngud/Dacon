
import numpy as np
import pandas as pd
import glob
import datetime
import cv2

import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dense, BatchNormalization, Activation, Reshape, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

es=EarlyStopping(
    patience=100,
    verbose=1
)

mc=ModelCheckpoint(
    'c:/data/modelcheckpoint/dacon2.hdf5',
    save_best_only=True,
    verbose=1
)

rl=ReduceLROnPlateau(
    patience=30,
    verbose=1,
    factor=0.5
)

# train_list=glob.glob('c:/data/dacon/data2/dirty_mnist_train/*.png')
# test_list=glob.glob('c:/data/dacon/data2/dirty_mnist_test/*.png')
# answer_csv=pd.read_csv('c:/data/dacon/data2/dirty_mnist_answer.csv', index_col=0, header=0)

# train_numpy=list()
# test_numpy=list()

# for i in train_list:
#     img=cv2.imread(i, cv2.IMREAD_GRAYSCALE)
#     img=cv2.resize(img, (128, 128))
#     img=np.array(img)/255.
#     train_numpy.append(img)


# for i in test_list:
#     img=cv2.imread(i, cv2.IMREAD_GRAYSCALE)
#     img=cv2.resize(img, (128, 128))
#     img=np.array(img)/255.
#     test_numpy.append(img)

# print(len(train_numpy))
# print(train_numpy[0])
# print(train_numpy[0].shape)

# train_list=np.array(train_numpy)
# test_list=np.array(test_numpy)
# answer_list=answer_csv.to_numpy()

# plt.imshow(train_numpy[0])
# plt.show()

# np.save('c:/data/dacon/data2/train.npy', arr=train_numpy)
# np.save('c:/data/dacon/data2/test.npy', arr=test_numpy)
# np.save('c:/data/dacon/data2/answer.npy', arr=answer_list)


# npy 로드
train=np.load('c:/data/dacon/data2/train.npy') # x
answer=np.load('c:/data/dacon/data2/answer.npy') # y
test=np.load('c:/data/dacon/data2/test.npy')

kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=23
)

# print(train.shape) # (50000, 128, 128)
# print(answer.shape) # (50000, 26)
# print(test.shape) # (5000, 128, 128)

x_train, x_val, y_train, y_val=train_test_split(
    train, answer,
    train_size=0.8,
    random_state=23
)

x_train, x_test, y_train, y_test=train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=23
)

# print(x_train.shape) # (40000, 128, 128)
# print(y_train.shape) # (10000, 128, 128)
# print(x_val.shape)
# print(x_test.shape)

# # 데이터 분리

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen2=ImageDataGenerator()

x_train=x_train.reshape(-1, 128, 128, 1)
x_val=x_val.reshape(-1, 128, 128, 1)
x_test=x_test.reshape(-1, 128, 128, 1)
test=test.reshape(-1, 128, 128, 1)

batch=16

train_set=datagen.flow(
    x_train, y_train,
    batch_size=batch)

test_set=datagen2.flow(
    x_test, y_test
)

val_set=datagen2.flow(
    x_val, y_val
)

test_gen=datagen2.flow(
    test
)

model=Sequential()
model.add(Conv2D(32, 2, padding='same', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(52))
model.add(Reshape((26, 2)))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.1),
    loss='sparse_categorical_crossentropy',
    metrics='acc'
)

# model.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     batch_size=4,
#     epochs=1
# )

model.fit(
    train_set,
    validation_data=val_set,
    steps_per_epoch=len(x_train)//batch,
    epochs=5,
    callbacks=[es, mc, rl]
)

# loss=model.evaluate(
#     x_test, y_test
# )

loss=model.evaluate(
    test_set
)

# pred=model.predict(
#     x_test
# )

model.load_weights('c:/data/modelcheckpoint/dacon2.hdf5')

pred=model.predict(
    test_gen
)

print(np.argmax(pred[:5], axis=-1))

print(loss)


# print(y_test[:5])

# results
# try : [0.6907089948654175, 0.5378028750419617]
# try 2: [0.6905454993247986, 0.5378028750419617]

# [7.858035564422607, 0.5124711394309998]
# [8.158544540405273, 0.4938269257545471]