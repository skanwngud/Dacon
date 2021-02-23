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

# c:/data/dacon/data2/dirty_mnist/
# c:/data/dacon/data2/test_dirty_mnist
# c:/data/dacon/data2/dirty_mnist_answer.csv


kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=23
)

# 이미지 로드 / npy 저장

str_time=datetime.datetime.now()

train_list=glob.glob('c:/data/dacon/data2/dirty_mnist/*.png')
test_list=glob.glob('c:/data/dacon/data2/test_dirty_mnist/*.png')
answer_csv=pd.read_csv('c:/data/dacon/data2/dirty_mnist_answer.csv', index_col=0, header=0)

df_train=pd.read_csv('c:/data/dacon/data2/mnist_data/train.csv', index_col=0, header=0)
dftrain=df_train.iloc[:, 2:].values # mnist1 x
dfanswer=df_train.iloc[:, 1].values # mnist1 y

# digit OneHotEncoding
label=list()
for i in range(len(dfanswer)):
    if dfanswer[i]=='A':
        digit=0
        label.append(digit)
    elif dfanswer[i]=='B':
        digit=1
        label.append(digit)
    elif dfanswer[i]=='C':
        digit=2
        label.append(digit)
    elif dfanswer[i]=='D':
        digit=3
        label.append(digit)
    elif dfanswer[i]=='E':
        digit=4
        label.append(digit)
    elif dfanswer[i]=='F':
        digit=5
        label.append(digit)
    elif dfanswer[i]=='G':
        digit=6
        label.append(digit)
    elif dfanswer[i]=='H':
        digit=7
        label.append(digit)
    elif dfanswer[i]=='I':
        digit=8
        label.append(digit)
    elif dfanswer[i]=='J':
        digit=9
        label.append(digit)
    elif dfanswer[i]=='K':
        digit=10
        label.append(digit)
    elif dfanswer[i]=='L':
        digit=11
        label.append(digit)
    elif dfanswer[i]=='M':
        digit=12
        label.append(digit)
    elif dfanswer[i]=='N':
        digit=13
        label.append(digit)
    elif dfanswer[i]=='O':
        digit=14
        label.append(digit)
    elif dfanswer[i]=='P':
        digit=15
        label.append(digit)
    elif dfanswer[i]=='Q':
        digit=16
        label.append(digit)
    elif dfanswer[i]=='R':
        digit=17
        label.append(digit)
    elif dfanswer[i]=='S':
        digit=18
        label.append(digit)
    elif dfanswer[i]=='T':
        digit=19
        label.append(digit)
    elif dfanswer[i]=='U':
        digit=20
        label.append(digit)
    elif dfanswer[i]=='V':
        digit=21
        label.append(digit)
    elif dfanswer[i]=='W':
        digit=22
        label.append(digit)
    elif dfanswer[i]=='X':
        digit=23
        label.append(digit)
    elif dfanswer[i]=='Y':
        digit=24
        label.append(digit)
    elif dfanswer[i]=='Z':
        digit=25
        label.append(digit)
    else:
        pass

label=to_categorical(label)

print(label)

# print(len(train_list)) # 50000
# print(len(test_list)) # 5000
# print(train_list_numpy.shape) # (2048, 28, 28)
# print(train_list_answer[0]) # L
# print(train_list_answer.shape) # (2048, )

# img2=cv2.imread(train_list_numpy[0], cv2.IMREAD_GRAYSCALE)
# img2=cv2.resize(img2, (128, 128))

x=dftrain
y=label

x=x.reshape(-1, 28, 28, 1)/255.

print(y.shape)

x_train, x_val, y_train, y_val=train_test_split(
    x, y,
    train_size=0.8,
    random_state=23
)

x_train, x_test, y_train, y_test=train_test_split(
    x_train, y_train,
    train_size=0.9,
    random_state=23
)

print(x_test)
print(y_test)

# print(x_train.shape) # (1474, 28, 28, 1)
# print(x_val.shape) # (410, 28, 28, 1)
# print(x_test.shape) # (164, 28, 28, 1)
# print(y_train.shape) # (1474, )

model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(26, activation='softmax'))

es=EarlyStopping(
    patience=100,
    verbose=1
)

mc=ModelCheckpoint(
    'c:/data/modelcheckpoint/dacon2.hdf5',
    verbose=1,
    save_best_only=True
)

rl=ReduceLROnPlateau(
    verbose=1,
    factor=0.5,
    patience=50
)

model.compile(
    optimizer=Adam(
        learning_rate=0.1),
    loss='categorical_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    epochs=1000,
    callbacks=[es, mc, rl]
)

loss=model.evaluate(
    x_test, y_test
)

pred=model.predict(
    x_test
)

pred=np.argmax(pred)

print(y_test.shape)
print(x_test.shape)

print(loss)
print(pred)
print(pred[0])
# print(y_test[:5])

'''
train_numpy=list()
test_numpy=list()

for i in train_list:
    img=cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img, (128, 128))
    img=np.array(img)/255.
    train_numpy.append(img)


for i in test_list:
    img=cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img, (128, 128))
    img=np.array(img)/255.
    test_numpy.append(img)

print(len(train_numpy))
print(train_numpy[0])
print(train_numpy[0].shape)

train_list=np.array(train_numpy)
test_list=np.array(test_numpy)
answer_list=answer_csv.to_numpy()

plt.imshow(train_numpy[0])
plt.show()

np.save('c:/data/dacon/data2/train.npy', arr=train_numpy)
np.save('c:/data/dacon/data2/test.npy', arr=test_numpy)
np.save('c:/data/dacon/data2/answer.npy', arr=answer_list)


# npy 로드
train=np.load('c:/data/dacon/data2/train.npy') # x
answer=np.load('c:/data/dacon/data2/answer.npy') # y
test=np.load('c:/data/dacon/data2/test.npy')



# print(train.shape) # (50000, 128, 128)
# print(answer.shape) # (50000, 26)
# print(test.shape) # (5000, 128, 128)

# x_train, x_val, y_train, y_val=train_test_split(
#     train, answer,
#     train_size=0.8,
#     random_state=23
# )

# x_train, x_test, y_train, y_test=train_test_split(
#     x_train, y_train,
#     train_size=0.8,
#     random_state=23
# )

# print(x_train.shape) # (40000, 128, 128)
# print(y_train.shape) # (10000, 128, 128)
# print(x_val.shape)
# print(x_test.shape)

# # 데이터 분리

for train_index, val_index in kf.split(train, answer):
    x_train=train[train_index]
    y_train=answer[train_index]
    x_val=train[val_index]
    y_val=answer[val_index]

    x_train, x_test, y_train, y_test=train_test_split(
        x_train, y_train,
        train_size=0.9,
        random_state=23
    )

    x_train=x_train.reshape(-1, 128, 128, 1)
    x_val=x_val.reshape(-1, 128, 128, 1)
    x_test=x_test.reshape(-1, 128, 128, 1)

    model=Sequential()
    model.add(Conv2D(32, 2, padding='same', input_shape=(128, 128, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(52))
    model.add(Reshape((26, 2)))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics='acc'
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=6,
        epochs=1
    )

    loss=model.evaluate(
        x_test, y_test
    )

    pred=model.predict(
        x_test
    )
    print(np.argmax(pred[:5], axis=-1))

    print(loss)
# print(y_test[:5])

# results
# try : [0.6907089948654175, 0.5378028750419617]
# try 2: [0.6905454993247986, 0.5378028750419617]
'''