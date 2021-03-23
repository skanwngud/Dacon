import numpy as np
import pandas as pd
import tensorflow
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,\
    BatchNormalization, Activation, Dropout
from tensorflow.keras.applications import MobileNet, EfficientNetB4
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split, KFold

test = list()
for i in range(72000):
    temp = cv2.imread(
        'c:/LPD_competition/test/%s.jpg'%i,
    )
    temp = cv2.resize(temp, (200, 200))
    temp = np.array(temp)
    test.append(temp)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

train = list()
label = list()
for i in range(1000):
    for j in range(48):
        temp = cv2.imread(
            'c:/LPD_competition/train/' + str(i) +'/'+ str(j) + '.jpg',
        )
        temp = cv2.resize(temp, (200, 200))
        temp = np.array(temp)
        train.append(temp)
        label.append(i)

test = np.array(test)
train = np.array(train)
label = np.array(label)

print(test.shape)
print(train.shape)
print(label.shape)

print(label[:50])

np.save(
    'c:/data/npy/lotte_x_2.npy', arr = train
)

np.save(
    'c:/data/npy/lotte_y_2.npy', arr = label
)

np.save(
    'c:/data/npy/lotte_test_2.npy', arr = test
)