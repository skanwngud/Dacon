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

train=np.load('C:/data/dacon/data2/train_set.npy')
test=np.load('C:/data/dacon/data2/test_set.npy')
ans=pd.read_csv('C:/data/dacon/data2/dirty_mnist_answer.csv', index_col=0)

print(datetime.datetime.now()-str_time)

ans=ans.to_numpy()


# train=train.reshape(-1, 256, 256, 3)

plt.imshow(train[0])
plt.imshow(test[0])
plt.show()

print(train[0])
print(train[0].shape) # (256, 256, 1)
print(ans[0])
print(ans[0].shape)

ans=to_categorical(ans)
ans=ans.reshape(-1, 26, 1, 2)

print(ans[0])
print(ans.shape) # (50000, 26,1, 2)
