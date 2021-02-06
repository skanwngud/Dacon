import numpy as np
import pandas as pd

import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

train=np.load('../data/dacon/data2/image_set.npy', encoding='ASCII')
ans=pd.read_csv('../data/dacon/data2/dirty_mnist_2nd_answer.csv', index_col=0)

print(train.shape)
print(ans.info())

anss=ans.to_numpy()

anss=to_categorical(anss)

train=train.reshape(-1, 256, 256, 1)
# anss=anss.reshape(50000*26*2)

print(anss.shape)
print(anss[0])

datagen=ImageDataGenerator(width_shift_range=(-1, 1), height_shift_range=(-1, 1))
datagen2=ImageDataGenerator()

# kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

# for train_index, test_index in kf.split(train, anss):
#     x_train=train[train_index]
#     x_test=train[test_index]
#     y_train=anss[train_index]
#     y_test=anss[test_index]

x_train, x_test, y_train, y_test=train_test_split(train, anss,
            train_size=0.9, random_state=23)

trainsets=datagen.flow(x_train, y_train, batch_size=256)
testsets=datagen2.flow(x_test, y_test)
# answer=datagen2.flow(anss, shuffle=False)

model=Sequential()
model.add(Conv2D(32, 2, padding='same', input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Flatten)
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(26, activation='softmax'))

es=EarlyStopping(patience=50, verbose=1)
rl=ReduceLROnPlateau(patience=10, verbose=1, factor=0.5)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/vision2_{val_acc:.4f}-{val_loss:.4f}.hdf5',
                verbose=1, save_best_only=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit_generator(trainsets, validation_data=(testsets),
        steps_per_epoch=256, epochs=100, callbacks=[es, cp, rl])

model.predict_generator(answer)

