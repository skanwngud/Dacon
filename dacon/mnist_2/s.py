import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape,\
    BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator as img

import datetime
import gc

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder

str_time=datetime.datetime.now()

datagen=img(width_shift_range=(-1, 1), height_shift_range=(-1, 1))
datagen2=img()

image_set=np.load('../data/dacon/data2/image_set.npy')
predict=np.load('../data/dacon/data2/test_set.npy')

answer=pd.read_csv('../data/dacon/data2/dirty_mnist_2nd_answer.csv', index_col=0)

answer=answer.to_numpy()

print(image_set.shape) # (50000, 256, 256)
# print(answer.info()) # (50000, 27)

answer=answer.reshape(-1, 26)

one=OneHotEncoder()
answer=one.fit_transform(answer).toarray()
answer=answer.reshape(-1, 26, 2)

image_set=image_set.reshape(-1, 256, 256, 1)/255
image_set=image_set.astype('float32')
predict=predict.reshape(-1, 256, 256, 1)/255
predict=predict.astype('float32')

print(answer.shape) # (50000, 26)
print(datetime.datetime.now()-str_time)

kf=KFold(n_splits=5, shuffle=True, random_state=23)

# for train_index, val_index in kf.split(image_set, answer):
#     x_train=image_set[train_index]
#     x_val=image_set[val_index]
#     y_train=answer[train_index]
#     y_val=answer[val_index]

x_train, x_test, y_train, y_test=train_test_split(image_set, answer,
                            train_size=0.9, random_state=23)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.9, random_state=23)
trainsest=datagen.flow(x_train, y_train, batch_size=32)
valsets=datagen2.flow(x_val, y_val)
predict=datagen2.flow(predict, shuffle=False)

# print(x_train.shape) # 36000, 256, 256
# print(x_val.shape) # 4000, 256,2 56
# print(x_test.shape) # 10000, 256, 256
# print(y_train.shape) # (36000, 26, 2)
# print(y_val.shape) # (4000, 26, 2)
# print(y_test.shape) # (10000, 26, 2)

model=Sequential()
model.add(Conv2D(32, 2, padding='same', input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(52))
model.add(Reshape((26, 2)))
model.add(Dense(2, activation='softmax'))

es=EarlyStopping(monitor='val_loss', patience=120, verbose=1)
rl=ReduceLROnPlateau(patience=50, verbose=1, factor=0.2 )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#             batch_size=32)
model.fit_generator(trainsest, validation_data=(valsets),
            steps_per_epoch=32, epochs=10000, callbacks=[es, rl])

loss=model.evaluate(x_test, y_test)
# pred=model.predict(predict)
pred=model.predict_generator(predict)
pred=np.where(pred>0.5, 1, 0)
pred=pred[:, :, 0]
gc.collect()

print(pred[:5])
print(pred.shape)
print(loss)
print(datetime.datetime.now()-str_time)

# sub=pd.read_csv('../data/dacon/data2/sample_submission.csv')
df_pred=pd.DataFrame(pred)
df_pred.to_csv('../data/dacon/data2/ss.csv')

print(datetime.datetime.now()-str_time)