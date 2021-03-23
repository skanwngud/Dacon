import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import cv2
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras.applications import MobileNet, EfficientNetB4
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,\
<<<<<<< HEAD
    BatchNormalization, Activation, Dense, Dropout, Input, Concatenate, \
        GlobalAveragePooling2D, Dropout
=======
    BatchNormalization, Activation, Dense, Dropout, Input, Concatenate, GlobalAveragePooling2D, GaussianDropout
>>>>>>> ec6166b03fcb5d13ec6e554a2de91f868e57224c
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import train

str_time = datetime.datetime.now()

<<<<<<< HEAD
datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=(-1, 1),
    height_shift_range=(-1, 1),
    rescale = 1./255
)
=======
# datagen = ImageDataGenerator(
#     vertical_flip=True,
#     horizontal_flip=True,
#     rotation_range=0.1,
#     width_shift_range=(-1, 1),
#     height_shift_range=(-1, 1),
#     validation_split=0.8
# )
>>>>>>> ec6166b03fcb5d13ec6e554a2de91f868e57224c

datagen2 = ImageDataGenerator(
    rescale=1./255
)

es = EarlyStopping(
    patience=50,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=10,
    verbose=1
)

mc = ModelCheckpoint(
    'c:/data/modelcheckpoint/lotte.hdf5',
    save_best_only=True,
    verbose=1
)

x = np.load(
    'c:/data/npy/lotte_x_2.npy'
)

y = np.load(
    'c:/data/npy/lotte_y_2.npy'
)

test = np.load(
    'c:/data/npy/lotte_test_2.npy'
)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

# mob = MobileNet(
#     include_top=False,
#     input_shape=(128, 128, 3)
# )

eff = EfficientNetB4(
    include_top=False,
    input_shape=(200, 200, 3)
)

# mob.trainable = True
eff.trainable = True

batch_size = 16

x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=0.8, random_state=23
)

train_set = datagen.flow(
    x_train, y_train,
    seed = 23,
    batch_size = batch_size
)

val_set = datagen2.flow(
    x_val, y_val,
    seed = 23,
    batch_size = batch_size
)

epochs = len(x_train)//batch_size

# 학원에서는 x,test 에 /255. 하고 집에서는 /255. 하지 말 것

model = Sequential()
model.add(eff)
<<<<<<< HEAD
model.add(GlobalAveragePooling2D())
model.add(Dense(4048, activation = 'swish'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation = 'softmax'))
=======
# model.add(Conv2D(1024, kernel_size=3, padding='same', activation = 'swish'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='swish'))
model.add(GaussianDropout(0.4))
model.add(Dense(1000, activation='softmax'))
>>>>>>> ec6166b03fcb5d13ec6e554a2de91f868e57224c

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics='acc'
)

hist = model.fit(
    train_set,
    validation_data=val_set,
    epochs=1000,
    steps_per_epoch=epochs,
    callbacks=[es, rl, mc],
    batch_size = batch_size
)

model.load_weights(
    'c:/data/modelcheckpoint/lotte.hdf5'
)

pred = model.predict(
    test
)

submission['prediction'] = np.argmax(pred, axis = -1)
submission.to_csv(
    'c:/data/csv/lotte.csv',
    index = False
)

<<<<<<< HEAD
print('time : ', datetime.datetime.now() - str_time)
=======
print(datetime.datetime.now() - str_time)
>>>>>>> ec6166b03fcb5d13ec6e554a2de91f868e57224c
print('done')