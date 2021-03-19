import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,\
    BatchNormalization, Activation, MaxPooling2D

from sklearn.model_selection import train_test_split

eff = EfficientNetB7(
    include_top=False,
    input_shape=(128, 128, 3)
)

es = EarlyStopping(
    patience=10,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=5,
    verbose=1
)

mc = ModelCheckpoint(
    'c:/data/modelcheckpoint/lotte.hdf5',
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

train_set = datagen.flow_from_directory(
    'c:/LPD_competition/train/',
    subset = 'training',
    class_mode='categorical',
    target_size=(128, 128),
    batch_size=16
)

val_set = datagen.flow_from_directory(
    'c:/LPD_competition/train/',
    subset = 'validation',
    class_mode='categorical',
    target_size=(128, 128),
    batch_size=16
)

# test_set = datagen2.flow_from_directory(
#     'c:/LPD_competition/test_1/',
#     shuffle=False,
#     batch_size=16
# )

test_set = np.load(
    'c:/LPD_competition/pred.npy'
)
test_set = test_set.reshape(-1, 128, 128, 3)/255.

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

x_train = train_set[0][0]
y_train = train_set[0][1]
x_val = val_set[0][0]
y_val = val_set[0][1]

# print(train_set.shape)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

model = Sequential()
model.add(eff)
model.add(Conv2D(128, 3, padding='same'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1000, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    steps_per_epoch=len(train_set)//16,
    callbacks=[es, rl, mc]
)

loss = model.evaluate(
    x_val, y_val
)

model.load_weights(
    'c:/data/modelcheckpoint/lotte.hdf5'
)

pred = model.predict(
    test_set
)

print(loss)
print(pred[:5])

submission['prediction']=np.argmax(pred, axis=-1)
submission.to_csv(
    'c:/LPD_competition/submission.csv',
    index = False
)