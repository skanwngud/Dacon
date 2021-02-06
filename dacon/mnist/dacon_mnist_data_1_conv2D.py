## Conv2D, train_test_split


# import libraries
import numpy as np
import pandas as pd

# read csv files
train=pd.read_csv('../data/dacon/data/train.csv')
test1=pd.read_csv('../data/dacon/data/test.csv')
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

# print(train.info()) # (2408, 783)
# print(pred.info()) # (20480, 784)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape,\
        Dense, BatchNormalization, Activation, concatenate, Concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

x=train.drop(['id', 'letter', 'digit'], axis=1) # Letter 제외
y=train['digit'].values # Digit 값
pred=test1.drop(['id', 'letter'], axis=1) # Letter 제외

x_letter=train['letter'].values # Letter 값
y_letter=train['letter'].values # Letter 값

pred_letter=test1['letter'].values # letter 값

# numpy 전환
x=x.to_numpy()
pred=pred.to_numpy()

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

x_l_train, x_l_test, y_l_train, y_l_test=train_test_split(x_letter, y_letter, train_size=0.8, random_state=23)

# 전처리
x_train=x_train.reshape(-1, 28, 28, 1)/255.
x_test=x_test.reshape(-1, 28, 28, 1)/255.
pred=pred/255.
pred_letter=pred_letter.reshape(-1, 1)

# OneHotEncoder 를 지나기 위한 형태변환
y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)
x_l_train=x_l_train.reshape(-1, 1)
x_l_test=x_l_test.reshape(-1, 1)
y_l_train=y_l_train.reshape(-1, 1)
y_l_test=y_l_test.reshape(-1, 1)

# 원핫인코딩
enc=OneHotEncoder()
y_train=enc.fit_transform(y_train).toarray() # 0 ~ 9 까지의 수를 구분하기 위함
y_test=enc.transform(y_test).toarray()
x_l_train=enc.fit_transform(x_l_train).toarray() # A~Z 까지의 글자를 구분하기 위함
x_l_test=enc.transform(x_l_test).toarray()

y_l_train=enc.fit_transform(y_l_train).toarray()
y_l_test=enc.transform(y_l_test).toarray()
# pred_letter=enc.transform(pred_letter).toarray()

x_l_train=x_l_train.reshape(-1, 26, 1, 1)
x_l_test=x_l_test.reshape(-1, 26, 1, 1)
y_l_train=y_l_train.reshape(-1, 26)
y_l_test=y_l_test.reshape(-1, 26)
pred_letter=pred_letter.reshape(-1, 1)

# cumsum=np.cumsum(pca.explained_variance_ratio_)
# print(np.argmax(cumsum>=0.95)+1) # 95

# print(x_train.shape) # (1310, 9)

predict=np.append(pred, pred_letter, axis=1)

# 최종 각 변수들의 형태
print(x_train.shape) # (1638, 28, 28, 1)
print(x_test.shape) # (410, 28, 28, 1)
print(x_l_train.shape) # (1638, 1, 1, 1)
print(x_l_test.shape) # (410, 1, 1, 1)

print(y_train.shape) # (1638, )
print(y_test.shape) # (410, )
print(y_l_train.shape) # (1638, 1)
print(y_l_test.shape) # (410, 1)

print(pred.shape) # (20480, 784)
print(pred_letter.shape) # (20480, 1)

print(predict.shape) # (20480, 785)

print(type(x_train))
print(type(x_l_train))
print(type(y_train))
print(type(y_l_train))



# model
input=Input(shape=(28, 28, 1))
conv2=Conv2D(64, 2, padding='same')(input)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2)(act)
conv2=Conv2D(128, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(256, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(512, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(1024, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(1024, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(512, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
flat=Flatten()(max)
dense=Dense(1000, activation='relu')(flat)
bn=BatchNormalization()(dense)
# output=Dense(10, activation='softmax')(bn)
# model=Model(input, output)

input2=Input(shape=(26, 1, 1))
conv22=Conv2D(64, 2, padding='same')(input2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(128, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(256, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(512, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(1024, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(1024, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(512, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
flat2=Flatten()(max2)
dense2=Dense(1000, activation='relu')(flat2)
bn2=BatchNormalization()(dense2)
# output2=Dense(26, activation='softmax')(bn2)
# model2=Model(input2, output2)

merge=concatenate([bn, bn2])
middle=Dense(256, activation='relu')(merge)
middle=Dense(10, activation='softmax')(middle)
model3=Model([input, input2], middle)


# compile
es=EarlyStopping(monitor='val_loss', patience=100, mode='auto')
rl=ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_2_{val_acc:.4f}-{val_loss:.4f}.hdf5',
                        monitor='val_acc', mode='auto', save_best_only=True)
cp2=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_22_{val_acc:.4f}-{val_loss:.4f}.hdf5',
                        monitor='val_acc', mode='auto', save_best_only=True)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# model.fit(x_train, y_train, validation_data=(x_test, y_test),
#         epochs=100, batch_size=128, callbacks=[es, rl, cp])
# model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# model2.fit(x_l_train, y_l_train, validation_data=(x_l_test, y_l_test),
#         epochs=100, batch_size=128, callbacks=[es, rl, cp2])

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model3.fit([x_train, x_l_train], [y_train, y_l_train], validation_data=([x_test, x_l_test], [y_test, y_l_test]),
                epochs=100, batch_size=128, callbacks=[es, rl])


# eval, pred
# loss=model.evaluate(x_test, y_test)
# y_pred=model.predict(pred)

# y_predict=pd.DataFrame(y_pred)
# y_predict.to_csv('../data/dacon/data/1.csv')

# print(loss)
# print(y_pred[0])
# print(pred[0])

# print(y_pred.shape)

# pred=pred.reshape(-1, 28, 28, 1)
sub['digit']=np.argmax(model3.predict(predict), axis=1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples.csv', index=False)
