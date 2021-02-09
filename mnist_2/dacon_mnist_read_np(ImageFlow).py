import numpy as np
import pandas as pd

import tensorflow

import datetime

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, Activation, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.optimizers import Adam

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold


str_time=datetime.datetime.now()

train=np.load('C:/data/dacon/data2/train_set.npy', encoding='ASCII')
test=np.load('C:/data/dacon/data2/test_set.npy', encoding='ASCII')
ans=pd.read_csv('C:/data/dacon/data2/dirty_mnist_answer.csv', index_col=0)

train=np.resize(train, (50000, 128, 128, 1))
test=np.resize(test, (5000, 128, 128, 1))

print('file read : ', datetime.datetime.now()-str_time)

print(train.shape)
print(test.shape)
# print(ans.info())

anss=ans.to_numpy()

anss=to_categorical(anss)
# train=train.reshape(50000*26)
# anss=anss.reshape(50000*26)

# print(anss.shape) # (50000, 26, 2)

datagen=ImageDataGenerator(width_shift_range=(-1, 1), height_shift_range=(-1, 1))
datagen2=ImageDataGenerator()

kf=KFold(n_splits=15, shuffle=True, random_state=23)

# train=train.reshape(-1, 256, 256, 1)
# test=test.reshape(-1, 256, 256, 1)
# anss=anss.reshape(-1, 26, 1)

print('train start : ', datetime.datetime.now()-str_time)

i=0
results=list()
for train_index, test_index in kf.split(train, anss):
    x_train=train[train_index]
    x_test=train[test_index]
    y_train=anss[train_index]
    y_test=anss[test_index]

    x_train, x_test, y_train, y_test=train_test_split(train, anss,
                train_size=0.9, random_state=23)

    trainsets=datagen.flow(x_train, y_train, batch_size=50)
    testsets=datagen2.flow(x_test, y_test)
    # answer=datagen2.flow(anss, shuffle=False)

    i+=1
    print(str(i) + ' 번째 훈련')
    # print(test.shape) # (5000, 256, 256, 1)
    # print(x_train.shape) # (45000, 256, 256, 1)
    # print(y_train.shape) # (45000, 1, 26)
    # print(x_test.shape) # (5000, 256, 256, 1)
    # print(y_test.shape) # (5000, 26, 2)

    # print(anss.shape) # (50000, 1, 26)

    # y_train=y_train.reshape(-1, 52)
    # y_test=y_test.reshape(-1, 52)

    model=Sequential()
    model.add(Conv2D(32, 2, padding='same', input_shape=(128, 128, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(32, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(52))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((26, 2)))
    model.add(Dense(2, activation='softmax'))

    # model.summary()

    es=EarlyStopping(patience=50, verbose=1)
    rl=ReduceLROnPlateau(patience=20, verbose=1, factor=0.1)
    cp=ModelCheckpoint(filepath='../data/modelcheckpoint/vision2_%s_{val_acc:.4f}-{val_loss:.4f}.hdf5'%i,
                    verbose=1, save_best_only=True)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.1), metrics='acc')
    # model.fit(x_train, y_train, validation_data=(x_test, y_test),
    #         batch_size=32, epochs=1, callbacks=[es, cp, rl])
    model.fit(trainsets, validation_data=(testsets),
                epochs=1000, steps_per_epoch=900 ,callbacks=[es, rl, cp])

    # pred=model.predict(test)
    pred=model.predict_generator(testsets)
    # results+=pred/5

    # predict=np.argmax(pred, axis=-1)
    predict=np.where(pred>0.5, 1, 0)
    predict=predict[:, 0]

    print(pred[0])
    print(predict[0])
    print(predict.shape) # (5000, 2)

    # predict=predict.reshape(-1, 26*2)

    print(str(i) + ' 번째 훈련 종료')

print('train finish : ', datetime.datetime.now()-str_time)

predict=pd.DataFrame(predict)

# sub=pd.read_csv('../data/dacon/data2/sample_submission.csv')

sub=predict.to_csv('../data/dacon/data2/submission.csv', index=False)
