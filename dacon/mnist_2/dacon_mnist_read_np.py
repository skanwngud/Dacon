import numpy as np
import pandas as pd

import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, Activation, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

train=np.load('../data/dacon/data2/image_set.npy', encoding='ASCII')
test=np.load('../data/dacon/data2/test_set.npy', encoding='ASCII')
ans=pd.read_csv('../data/dacon/data2/dirty_mnist_2nd_answer.csv', index_col=0)

# print(train.shape)
# print(ans.info())

anss=ans.to_numpy()

anss=to_categorical(anss)
# train=train.reshape(50000*26)
# anss=anss.reshape(50000*26)

# print(anss.shape) # (50000, 26, 2)

datagen=ImageDataGenerator(width_shift_range=(-1, 1), height_shift_range=(-1, 1))
datagen2=ImageDataGenerator()

kf=KFold(n_splits=5, shuffle=True, random_state=23)

train=train.reshape(-1, 256, 256, 1)
test=test.reshape(-1, 256, 256, 1)
# anss=anss.reshape(-1, 26, 2)

# results=list()
# for train_index, test_index in kf.split(train, anss):
#     x_train=train[train_index]
#     x_test=train[test_index]
#     y_train=anss[train_index]
#     y_test=anss[test_index]

x_train, x_test, y_train, y_test=train_test_split(train, anss,
            train_size=0.9, random_state=23)

# trainsets=datagen.flow(x_train, y_train, batch_size=16)
# testsets=datagen2.flow(x_test, y_test)
# answer=datagen2.flow(anss, shuffle=False)


print(test.shape) # (5000, 256, 256, 1)
print(x_train.shape) # (45000, 256, 256, 1)
print(y_train.shape) # (45000, 1, 26)
print(x_test.shape) # (5000, 256, 256, 1)
print(y_test.shape) # (5000, 26, 2)

print(anss.shape) # (50000, 1, 26)

# y_train=y_train.reshape(-1, 52)
# y_test=y_test.reshape(-1, 52)

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
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(52))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Reshape((26, 2)))
model.add(Dense(26, activation='softmax'))

model.summary()

es=EarlyStopping(patience=50, verbose=1)
rl=ReduceLROnPlateau(patience=10, verbose=1, factor=0.5)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/vision2_{val_acc:.4f}-{val_loss:.4f}.hdf5',
                verbose=1, save_best_only=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_test, y_test),
        batch_size=32, epochs=1, callbacks=[es, cp, rl])
# model.fit(trainsets, validation_data=(testsets),
#             epochs=5, steps_per_epoch=16 ,callbacks=[es, rl, cp])

pred=model.predict(test)
# pred=model.predict(testsets)
# results+=pred/5

predict=np.argmax(pred, axis=-1)

print(pred[0])
print(predict[0])
print(predict.shape) # (5000, 2)

predict=predict.reshape(-1, 26*2)

predict=pd.DataFrame(predict)

# sub=pd.read_csv('../data/dacon/data2/sample_submission.csv')

sub=predict.to_csv('../data/dacon/data2/submission.csv', index=False)

# tensorflow.python.framework.errors_impl.InvalidArgumentError:  Incompatible shapes: [32,1,26] vs. [32] no Reshape
# tensorflow.python.framework.errors_impl.InvalidArgumentError:  logits and labels must have the same first dimension, got logits shape [32,26] and labels shape [1664] # 1, 26 Reshape