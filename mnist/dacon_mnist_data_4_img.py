import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from sklearn.model_selection import train_test_split, KFold

train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
test=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

x_image=train.iloc[:, 2:].values.reshape(-1, 28, 28, 1) # letter 제외
x_letter=train.letter # letter 행만 선택
y=to_categorical(train.digit) # digit 행만 선택

print(x_image.shape) # (2048, 28, 28, 1)
print(x_letter.shape) # (2048, )
print(y.shape) # (2048, 10)

datagen=ImageDataGenerator(
    width_shift_range=5,
    height_shift_range=5,
    rotation_range=10,
    zoom_range=0.05
)

flow1=datagen.flow(x_image, x_letter, batch_size=32, seed=2021)
flow2=datagen.flow(x_image, y, batch_size=32, seed=2021)

x_image_gen1, x_letter_gen=flow1.next()
x_image_gen1, y_gen=flow2.next()

kf=KFold(n_splits=5, shuffle=True, random_state=22)

x_train, x_test, y_train, y_test=train_test_split(x_image, y,
                train_size=0.8, random_state=23)
x_train, x_test, x_letter_train, x_letter_test=train_test_split(x_image, x_letter,
                train_size=0.8, random_state=23)

print(type(x_image_gen1))
print(type(x_train))

x_train, x_letter_train=flow1.next()
x_train, y_train=flow2.next()

print('x_train.shape={}'.format(x_train.shape)) # (1638, 28, 28, 1) # (32, 28, 28, 1)
# print(x_letter_train.shape) # (1638, )
# print(y_train.shape) # (1638, 10)

x_letter_train=x_letter_train.reshape(-1, 1, 1, 1)

model=Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_letter_train, y_train),
            epochs=100, batch_size=128)

model.predict(test)

# 모르겠당