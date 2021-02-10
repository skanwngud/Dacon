import numpy as np
import pandas as pd

# train=np.load('c:/data/dacon/data2/train_set.npy')
# test=np.load('c:/data/dacon/data2/test_set.npy')
answer=pd.read_csv('c:/data/dacon/data2/dirty_mnist_answer.csv', index_col=0, header=0)

import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout,\
    Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=0.2
)
datagen2=ImageDataGenerator()

train_set=datagen.flow_from_directory(
    'c:/data/dacon/data2/dirty_mnist',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=10,
    subset='training'
)

test_set=datagen.flow_from_directory(
    'c:/data/dacon/data2/dirty_mnist',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=10,
    subset='validation'
)

pred_set=datagen2.flow_from_directory(
    'c:/data/dacon/data2/test_dirty_mnist',
    target_size=(128, 128),
    batch_size=10,
    class_mode='categorical'
)

ans=answer.to_numpy()

print(train_set[0][0].shape)
print(train_set[0][1].shape)
print(test_set[0].shape)
print(ans.shape)