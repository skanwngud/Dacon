# model ensemble
# 데이터 사이즈 안 맞아서 안 돌아감

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

x2 = np.load(
    'c:/data/npy/lotte_x.npy'
)

y2 = np.load(
    'c:/data/npy/lotte_y.npy'
)

test2 = np.load(
    'c:/data/npy/lotte_test.npy'
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

x2_set = datagen.flow(
    x2,
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

x2_train, x2_val, y2_train, y2_val = train_test_split(
    x2, y2, train_size = 0.8, random_state = 23
)

# 학원에서는 x,test 에 /255. 하고 집에서는 /255. 하지 말 것

input1 = Input(shape = (128, 128, 1))
a = Conv2D(128, 3, padding='same')(input1)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = MaxPooling2D(3, padding='same')(a)
a = Conv2D(256, 3, padding='same')(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = MaxPooling2D(3, padding='same')(a)
a = Conv2D(512, 3, padding='same')(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = MaxPooling2D(3, padding='same')(a)
a = Flatten()(a)
a = Dense(128)(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = Dense(256)(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = Dense(512)(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
output1 = Dense(1024, activation='relu')(a)

input2 = Input(shape = (128, 128, 3))
b = Conv2D(128, 3, padding='same')(input1)
b = BatchNormalization()(b)
b = Activation('relu')(b)
b = MaxPooling2D(3, padding='same')(b)
b = Conv2D(256, 3, padding='same')(b)
b = BatchNormalization()(b)
b = Activation('relu')(b)
b = MaxPooling2D(3, padding='same')(b)
b = Conv2D(512, 3, padding='same')(b)
b = BatchNormalization()(b)
b = Activation('relu')(b)
b = MaxPooling2D(3, padding='same')(b)
b = Flatten()(b)
b = Dense(128)(b)
b = BatchNormalization()(b)
b = Activation('relu')(b)
b = Dense(256)(b)
b = BatchNormalization()(b)
b = Activation('relu')(b)
b = Dense(512)(b)
b = BatchNormalization()(b)
b = Activation('relu')(b)
output2 = Dense(1024, activation='relu')(b)

merge = Concatenate([output1, output2])
c = Dense(1024, activation='relu')(merge)
c = Dense(2048, activation='relu')(c)
c = Dense(1024, activation='relu')(c)
output = Dense(1000, activation='softmax')(c)


model = Model([input1, input2], output)

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics='acc'
)

hist = model.fit(
    [x_train, x2_train], [y_train, y2_train],
    validation_data=([x_val, x2_val],  [y_val, y2_val]),
    epochs=1,
    steps_per_epoch=epochs,
    callbacks=[es, rl, mc]
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

plt.plot(hist)
plt.show()