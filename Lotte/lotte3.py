import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras.applications import MobileNet, EfficientNetB4
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,\
    BatchNormalization, Activation, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es = EarlyStopping(
    patience=100,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=30,
    verbose=1
)

mc = ModelCheckpoint(
    'c:/data/modelcheckpoint/lotte.hdf5',
    save_best_only=True,
    verbose=1
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

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

mob = MobileNet(
    include_top=False,
    input_shape=(128, 128, 3)
)

eff = EfficientNetB4(
    include_top=False,
    input_shape=(128, 128, 3)
)

mob.trainable = True
eff.trainable = True

x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=0.8, random_state=23
)

model = Sequential()
model.add(eff)
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Dense(128, activation='relu'))
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

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
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

print(test[:5])
print(pred[:5])