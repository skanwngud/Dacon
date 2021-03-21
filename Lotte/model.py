import numpy as np
import pandas as pd

from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,\
    BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split, KFold

time_now = datetime.now()

kf = KFold(
    n_splits = 5
)

eff = EfficientNetB4(
    include_top=False,
    input_shape=(128, 128, 3)
)

es = EarlyStopping(
    patience=20,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=10,
    verbose=1
)


datagen = ImageDataGenerator(
    width_shift_range=(-1, 1),
    height_shift_range=(-1, 1),
    rotation_range=0.1,
    rescale=1./255,
    validation_split=0.2
)

datagen2 = ImageDataGenerator(
    rescale=1./255
)

x = np.load(
    'c:/data/npy/lotte_x.npy'
)

y = np.load(
    'c:/data/npy/lotte_y.npy'
)

test = np.load(
    'c:/data/npy/lotte_test.npy'
)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=0.8, random_state=23
)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

# print(train_set.shape)
# print(x_train.shape)
# print(y_train.shape)
# print(x_val.shape)
# print(y_val.shape)

mc = ModelCheckpoint(
    'c:/LPD_competition/lotte_.hdf5',
    verbose=1,
    save_best_only=True
)

model = Sequential()
model.add(eff)
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('swish'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('swish'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1000, activation='softmax'))

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs = 500,
    callbacks = [es, rl, mc],
    batch_size = 16
)


model.load_weights(
    'c:/LPD_competition/lotte_.hdf5'
)

pred = model.predict(
    test
)

results = model.predict(
    test
)
results += results


submission['prediction']=np.argmax(results, axis=-1)
submission.to_csv(
    'c:/LPD_competition/submission_.csv',
    index = False
)

print('total_time : ', datetime.now() - time_now)
print('done!')