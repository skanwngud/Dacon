# import library
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# import train, submission files
train=pd.read_csv('./dacon/train/train.csv')
submission=pd.read_csv('./dacon/sample_submission.csv')

'''
def Add_features(data):
    data['cos']=np.cos(np.pi/2-np.abs(data['Hour']%12-6)/6*np.pi/2)
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])
    data.drop(['cso'], axis=1, inplace=True)
    return data
'''

# add to TARGET1,2, preprocessing data ('Hour')
def preprocess_data(data, is_train=True):
    data['cos']=np.cos(np.pi/2-np.abs(data['Hour']%12-6)/6*np.pi/2)
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])
    temp=data.copy()
    temp=temp[['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True: # train dataset
        temp['TARGET1']=temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['TARGET2']=temp['TARGET'].shift(-96).fillna(method='ffill')
        temp=temp.dropna()
        return temp.iloc[:-96]

    elif is_train==False: # test dataset
        temp=temp[['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :]


df_train=preprocess_data(train) # train data 컬럼 추가 및 프리프로세싱
print(df_train.info())
x_train=df_train.to_numpy() # numpy 변환

df_test=list() # df_test 합친 파일을 받을 리스트 생성
for i in range(81): # 0~80.csv
    file_path='./dacon/test/%d.csv'%i # == './dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path) # import test file
    temp=preprocess_data(temp, is_train=False) # 
    df_test.append(temp)

x_test=pd.concat(df_test)
x_test=x_test.to_numpy() # numpy 변환
print(x_test.shape) # (648, )

print(x_train.shape) # (52464, 10) ('Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET1', 'TARGET2')
print(x_test.shape) # (648, )

def split_xy(data, time_steps): # time_steps : 몇 행 단위로 자를 것인지
    x=list() # x data 를 받을 리스트 생성
    y1=list() # 7일차 y data 를 받을 리스트 생성
    y2=list() # 8일차 y data 를 받을 리스트 생성
    for i in range(len(data)): # i 변수의 범위 설정
        x_end_number=i+time_steps
        if x_end_number>len(data):
            break
        tmp_x=data[i:x_end_number, :-2] # 맨 뒤의 TARGET1,2 를 뺀 컬럼 값 (TARGET1,2 는 7,8일차의 예측치)
        tmp_y1=data[x_end_number-1:x_end_number, -2] # TARGET1, 7일차 예측치
        tmp_y2=data[x_end_number-1:x_end_number, -1] # TARGET2, 8일차 예측치
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

x, y1, y2=split_xy(x_train, 1) # x_train data 를 한 행씩 자르고 x, y1, y2 로 나눔

print(x.shape) # (52464, 1, 8)
print(y1.shape) # (52464, 1)
print(y2.shape) # (52464, 1)

def split_x(data, time_steps):
    x=list() # x_test 를 받을 리스트 생성
    for i in range(len(data)):
        x_end_number=i+time_steps
        if x_end_number>len(data):
            break
        tmp_x=data[i:x_end_number]
        x.append(tmp_x)
    return np.array(x)

x_test=split_x(x_test, 1) # x_test data 를 한 행씩 자름
# x_test=x_test.reshape(-1, 1, 8)

# train test split 으로 훈련용 데이터 자르기
x_train, x_val, y1_train, y1_val, y2_train, y2_val=train_test_split(x, y1, y2, train_size=0.8, random_state=23)

# define quantile loss
def quantile_loss(q, y_true, y_pred):
    e=(y_true-y_pred) # quantile loss formula
    return K.mean(K.maximum(q*e, (q-1)*e), axis=1)

quantile=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# defin model
def models():
    model=Sequential()
    model.add(Conv1D(128, 2, padding='same', activation='relu', input_shape=(1, 8)))
    model.add(Conv1D(256, 2, padding='same', activation='relu'))
    model.add(Conv1D(128, 2, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model

# import callbacks
es=EarlyStopping(monitor='val_loss', patience=20, mode='min')
rl=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='min')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_model_last_{epoch:02d}-{val_loss:.4f}.hdf5',
                    save_best_only=True, monitor='val_loss', mode='min')

# compile, fitting
x=list()
for i in quantile:
    model=models()
    model.compile(loss=lambda y_true, y_pred:quantile_loss(i, y_true, y_pred),
                    optimizer='adam')
    model.fit(x_train, y1_train, validation_data=(x_val, y1_val),
                epochs=1000, batch_size=64, callbacks=[es, rl])
    pred=pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp1=pd.concat(x, axis=1)
df_temp1[df_temp1<0]=0
num_temp1=df_temp1.to_numpy()
submission.loc[submission.id.str.contains('Day7'), 'q_0.1':]=num_temp1

x=list()
for i in quantile:
    model=models()
    model.compile(loss=lambda y_true, y_pred:quantile_loss(i, y_true, y_pred),
                    optimizer='adam')
    model.fit(x_train, y2_train, validation_data=(x_val, y2_val),
                epochs=10, batch_size=64, callbacks=[es, rl])
    pred=pd.DataFrame(model.predict(x_test).round(2))
    x.append(pred)
df_temp2=pd.concat(x, axis=1)
df_temp2[df_temp2<0]=0
num_temp2=df_temp2.to_numpy()
submission.loc[submission.id.str.contains('Day8'), 'q_0.1':]=num_temp2

submission.to_csv('./dacon/submission_last.csv', index=False)

import matplotlib.pyplot as plt

ranges = 336
hours = range(ranges)
sub=submission[ranges:ranges+ranges]

q_01 = sub['q_0.1'].values
q_02 = sub['q_0.2'].values
q_03 = sub['q_0.3'].values
q_04 = sub['q_0.4'].values
q_05 = sub['q_0.5'].values
q_06 = sub['q_0.6'].values
q_07 = sub['q_0.7'].values
q_08 = sub['q_0.8'].values
q_09 = sub['q_0.9'].values

plt.figure(figsize=(18,2.5))
plt.subplot(1,1,1)
plt.plot(hours, q_01, color='red')
plt.plot(hours, q_02, color='#aa00cc')
plt.plot(hours, q_03, color='#00ccaa')
plt.plot(hours, q_04, color='#ccaa00')
plt.plot(hours, q_05, color='#00aacc')
plt.plot(hours, q_06, color='#aacc00')
plt.plot(hours, q_07, color='#cc00aa')
plt.plot(hours, q_08, color='#000000')
plt.plot(hours, q_09, color='blue')
plt.legend()
plt.show()