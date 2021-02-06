## Conv2D, KFold

#  import libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt

import datetime

# read csv files
train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
pred=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

# print(train.info()) # (2408, 783)
# print(pred.info()) # (20480, 784)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Dense, BatchNormalization, Activation
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.decomposition import PCA

datagen = ImageDataGenerator(width_shift_range=(-1, 1), height_shift_range=(-1, 1))

datagen2=ImageDataGenerator()

x=train.iloc[:, 2:] # Letter 제외
y=train.iloc[:, 0] # Digit 값
pred=pred.iloc[:, 1:] # Letter 제외

# numpy 전환
x=x.to_numpy()
y=y.to_numpy()
pred=pred.to_numpy()

print(x.shape) # (2048, 784)
print(y.shape) # (2048, )

x=x.reshape(-1, 28, 28, 1)/255.
pred=pred.reshape(-1, 28, 28, 1)/255.

# x=np.resize(56, 56)

# y=to_categorical(y)

n=50

kf=StratifiedKFold(n_splits=n, shuffle=True, random_state=22)

# print(pred.shape) # (20480, 28, 28, 1)

start_time=datetime.datetime.now()

i=0
result=0
val_loss_min=list()
for train_index, test_index in kf.split(x, y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

    i+=1
    print(str(n) + ' 번째 중 ' + str(i) + ' 번째 훈련')

    # x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.9, random_state=99)

    train=datagen.flow(x_train, y_train, batch_size=64)
    val=datagen2.flow(x_test, y_test)
    test=datagen2.flow(x_test, y_test)
    pred2=datagen2.flow(pred, shuffle=False)

    es=EarlyStopping(monitor='val_loss', patience=100, mode='auto')
    rl=ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', verbose=1, factor=0.1)
    cp=ModelCheckpoint(save_best_only=True, monitor='val_acc', mode='auto',
                        filepath='../data/modelcheckpoint/weight.h5', verbose=1)
    cp2=ModelCheckpoint(filepath='../data/modelcheckpoint/weight_%s_{val_acc:.4f}_{val_loss:.4f}.hdf5'%i,
                        save_best_only=True, monitor='val_acc', mode='auto')

    model=Sequential()
    model.add(Conv2D(256, 3, padding='same', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(1024, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(1024, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))

    # model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(epsilon=None), metrics='acc')
    # hist=model.fit(x_train, y_train, validation_data=(x_val, y_val),
    #             epochs=500, batch_size=128, callbacks=[es, rl, cp])
    hist=model.fit_generator(train, epochs=500, validation_data=(val),
                callbacks=[es, rl, cp, cp2], verbose=1)

    model.load_weights('../data/modelcheckpoint/weight.h5')
    result += model.predict_generator(pred2, verbose=True)/20

    # hists=pd.DataFrame(hist.history)
    # val_loss_min.append(hists['val_loss'].min)

    loss=model.evaluate_generator(test)

    print('loss : ', loss[0])
    print('acc : ', loss[1])

    print(str(n) + ' 번째 중 ' + str(i) + ' 번째 훈련 종료')
    
# sub['digit']=np.argmax(model.predict(pred), axis=1)

# sub['digit']=np.argmax(model.predict_generator(pred), axis=1)
sub['digit']=result.argmax(1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples.csv', index=False)

end_time=datetime.datetime.now()
spent_time=end_time-start_time

print('걸린 시간 : ', spent_time)