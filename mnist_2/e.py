import numpy as np
import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, \
    Reshape, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

# c:/data/dacon/data2/dirty_mnist_train
# c:/data/dacon/data2/drity_mnist_test
# c:/data/dacon/data2/drity_mnist_answer

'''
answer=pd.read_csv(
    'c:/data/dacon/data2/dirty_mnist_answer.csv',
    index_col=0, header=0)

# print(answer.info())

train=list()
for i in range(50000):
    img=cv2.imread(
        'c:/data/dacon/data2/dirty_mnist_train/%05d.png'%i,
        cv2.IMREAD_GRAYSCALE)
    # img=cv2.resize(img, (128, 128))
    img=np.array(img)/255.
    train.append(img)

test=list()
for i in range(50000, 55000):
    img=cv2.imread(
        'c:/data/dacon/data2/dirty_mnist_test/%d.png'%i,
        cv2.IMREAD_GRAYSCALE
    )
    img=np.array(img)/255.
    test.append(img)

train=np.array(train)
test=np.array(test)

print(train.shape)
print(test.shape)

x_train, x_val, y_train, y_val=train_test_split(
    train, answer,
    train_size=0.8,
    random_state=23
)

print(x_train.shape) # (40000, 256, 256)
print(y_train.shape) # (40000, 26)

np.save('c:/data/dacon/data2/x_train.npy', arr=x_train)
np.save('c:/data/dacon/data2/y_train.npy', arr=y_train)
np.save('c:/data/dacon/data2/x_val.npy', arr=x_val)
np.save('c:/data/dacon/data2/y_val.npy', arr=y_val)
np.save('c:/data/dacon/data2/test.npy', arr=test)
'''

x_train=np.load('c:/data/dacon/data2/x_train.npy')
y_train=np.load('c:/data/dacon/data2/y_train.npy')
x_val=np.load('c:/data/dacon/data2/x_val.npy')
y_val=np.load('c:/data/dacon/data2/y_val.npy')
test=np.load('c:/data/dacon/data2/test.npy')

print(x_train[0])

x_train=x_train.reshape(-1, 256, 256, 1)
x_val=x_val.reshape(-1, 256, 256, 1)

model=Sequential()
model.add(Conv2D(32, 2, padding='same', input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256))
model.add(Dense(26, activation='softmax'))

# model=model.load_weights('c:/data/modelcheckpoint/dacon2.hdf5')
model.compile(
    optimizer=Adam(
        learning_rate=0.01
    ),
    loss='sparse_categorica_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=1,
    verbose=1,
    validation_data=(x_val, y_val)
)

pred=model.predict(
    test
)

print(pred[:5])