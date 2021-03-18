import numpy as np
import pandas as pd
import tensorflow
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,\
    BatchNormalization, Activation, Dropout
from tensorflow.keras.applications import MobileNet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split, KFold

# test = np.load(
#     'c:/LPD_competition/test.npy'
# )

test = list()
for i in range(72000):
    temp = cv2.imread(
        'c:/LPD_competition/test/%s.jpg'%i
    )
    temp = cv2.resize(temp, (128, 128))
    temp = np.array(temp)/255.
    test.append(temp)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

batch_size = 16

es = EarlyStopping(
    patience=100,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=10,
    verbose=1,
    factor=0.5
)

mc = ModelCheckpoint(
    'c:/data/modelcheckpoint/lotte.hdf5',
    verbose=1,
    save_best_only=True
)

datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True,
    width_shift_range=(-1, 1),
    height_shift_range=(-1,1),
    validation_split=0.2
)

datagen2 = ImageDataGenerator()

train_set = datagen.flow_from_directory(
    'c:/LPD_competition/train/',
    target_size=(128, 128),
    subset='training',
    batch_size=batch_size
)

val_set = datagen.flow_from_directory(
    'c:/LPD_competition/train/',
    target_size=(128, 128),
    subset='validation',
    batch_size=batch_size
)

test_set = datagen2.flow(
    test,
    batch_size=batch_size,
    shuffle=False
)

x_train = train_set[0][0]
y_train = train_set[0][1]
x_val = val_set[0][0]
y_val = val_set[0][1]

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

epochs = len(train_set)//batch_size

eff = MobileNet(
    include_top=False,
    input_shape=(128, 128, 3)
)
eff.trainable=True
# modeling
model = Sequential()
model.add(eff)
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1000, activation='softmax'))

# compile, fitting
model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss='categorical_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs = 1000,
    steps_per_epoch=epochs,
    callbacks=[es, rl]
)

# predict
# model.load_weights(
#     'c:/data/modelcheckpoint/lotte.hdf5'
# )

pred = model.predict(
    test_set
)

submission['prediction']=np.argmax(pred, axis=-1)

submission.to_csv(
    'c:/LPD_competition/submission.csv',
    index=False
)