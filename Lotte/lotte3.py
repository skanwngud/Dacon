import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras.applications import MobileNet, EfficientNetB4
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,\
    BatchNormalization, Activation, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=0.1,
    width_shift_range=(-1, 1),
    height_shift_range=(-1, 1)
)

datagen2 = ImageDataGenerator()

es = EarlyStopping(
    patience=50,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=20,
    verbose=1
)

mc = ModelCheckpoint(
    'c:/data/modelcheckpoint/lotte.hdf5',
    save_best_only=True,
    verbose=1
)

x = np.load(
    'c:/data/npy/lotte_x_gr.npy'
)

y = np.load(
    'c:/data/npy/lotte_y_gr.npy'
)

test = np.load(
    'c:/data/npy/lotte_test_gr.npy'
)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

# mob = MobileNet(
#     include_top=False,
#     input_shape=(128, 128, 3)
# )

# eff = EfficientNetB4(
#     include_top=False,
#     input_shape=(128, 128, 3)
# )

# mob.trainable = True
# eff.trainable = True

x = x.reshape(-1, 128, 128, 1)
test = test.reshape(-1, 128, 128, 1)

batch_size = 16
epochs = len(x)//batch_size

x_set = datagen.flow(
    x,
    batch_size = batch_size
)

test_set = datagen2.flow(
    test,
    batch_size=batch_size,
    shuffle=False
)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=0.8, random_state=23
)

# 학원에서는 x,test 에 /255. 하고 집에서는 /255. 하지 말 것

model = Sequential()
# model.add(mob)
model.add(Conv2D(128, 3, padding='same', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(512, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Dense(512, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1000, activation='softmax'))

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics='acc'
)

hist = model.fit(
    x_train, y_train,
    validation_data=(x_val,  y_val),
    epochs=1000,
    steps_per_epoch=epochs,
    callbacks=[es, rl, mc],
    batch_size = batch_size
)

model.load_weights(
    'c:/data/modelcheckpoint/lotte.hdf5'
)

pred = model.predict(
    test_set
)

submission['prediction'] = np.argmax(pred, axis = -1)
submission.to_csv(
    'c:/data/csv/lotte.csv',
    index = False
)

print(np.argmax(test_set[:5], axis=-1))
print(np.argmax(pred[:5], axis=-1))