import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import cv2
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.applications import MobileNet, EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,\
    BatchNormalization, Activation, Dense, Dropout, Input, Concatenate, GlobalAveragePooling2D, GaussianDropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import train

str_time = datetime.datetime.now()

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=(-1, 1),
    height_shift_range=(-1, 1),
    # rescale = 1/255 # 쓰면 val_acc 박살남
)

datagen2 = ImageDataGenerator(
    # rescale= 1/255 # 마찬가지
)

es = EarlyStopping(
    patience=20,
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
    'c:/data/npy/lotte_xs.npy'
)

y = np.load(
    'c:/data/npy/lotte_ys.npy'
)

test = np.load(
    'c:/data/npy/lotte_tests.npy'
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
    input_shape=(128, 128, 3)
)

# mob.trainable = True
eff.trainable = True

batch_size = 32

x = preprocess_input(x)
test = preprocess_input(test)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=0.9, random_state=23
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
# model.add(Conv2D(1024, kernel_size=3, padding='same', activation = 'swish'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='swish'))
model.add(GaussianDropout(0.4))
model.add(Dense(1000, activation='softmax'))

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics='acc'
)

hist = model.fit_generator(
    train_set,
    validation_data=val_set,
    epochs=200,
    steps_per_epoch=1350,
    callbacks=[es, rl, mc]
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

print('time : ', datetime.datetime.now() - str_time)
print('done')