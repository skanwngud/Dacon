# import libraries
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, LSTM, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# define function
# GHI 컬럼 생성
def Add_feature(data):
    data['cos']=np.cos(np.pi/2-np.abs(data['Hour']%12-6)/6*np.pi/2)
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis=1, inplace=True)
    return data

# data split
def split_x(data, size):
    x=list()
    for i in range(len(data)-size+1):
        subset=data[i:(i+size)]
        x.append([item for item in subset])
    return np.array(x)

# define quantile_loss
def quantile_loss(q, y_true, y_pred):
    e=(y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e))

quantile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# GHI 컬럼 추가
def preprocess_data(data):
    data=Add_feature(data)
    temp=data.copy()
    temp=temp[['GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET']]
    return temp.iloc[:, :]

# 모델링
def models():
    model=Sequential()
    model.add(GRU(64, activation='relu' ,input_shape=(7, 7)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model

# 콜백 선언
es=EarlyStopping(monitor='val_loss', mode='auto', patience=20)
rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, factor=0.1)

# 컴파일
def compile(a, x_train, y_train, x_val, y_val):
    for q in quantile_list:
        print(str(q)+'번째')
        model=models()
        model.compile(loss=lambda y_true, y_pred:quantile_loss(q, y_true, y_pred),
                    optimizer='adam', metrics=['mae'])
        file_path=f'../data/modelcheckpoint/dacon_model_time{i}-{a}-{q}.hdf5'
        cp=ModelCheckpoint(file_path, save_best_only=True, monitor='val_loss')
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=100, batch_size=128, callbacks=[es, rl, cp], verbose=2)
    return model

# 예측
def predict(a, x_train, y_train, x_val, y_val, x_test):
    x=list()
    for q in quantile_list:
        model=load_model(f'../data/modelcheckpoint/dacon_model_time{i}-{a}-{q}.hdf5', compile=False)
        pred=pd.DataFrame(model.predict(x_test).round(2))
        x.append(pred)
    df_temp=pd.concat(x, axis=1)
    df_temp[df_temp<0]=0
    return df_temp

# 1. data
train=pd.read_csv('./dacon/train/train.csv')
sub=pd.read_csv('./dacon/sample_submission.csv')

test=list()
for i in range(81):
    a=pd.read_csv('./dacon/test/'+str(i)+'.csv')
    a=preprocess_data(a)
    test.append(a)
test=pd.concat(test)
test=np.array(test)

print(test.shape) # (27216, 7)

test=test.reshape(81, 7, 48, 7)
x_test=np.transpose(test, axes=(2,0,1,3))

data=train.values
data=data.reshape(1095, 48, 9)
data=np.transpose(data, axes=(1,0,2))

print(data.shape) # (48, 1095, 9)

data=data.reshape(48*1095, 9)
df=train.copy()
df.loc[:, :]=data
df.to_csv('c:/Study/dacon/df_train.csv', index=False)
train_data=preprocess_data(df)

es=EarlyStopping(monitor='val_loss', patience=20)
rl=ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

b=list()
c=list()
test=list()
for i in range(48):
    train_sort=train_data[1095*(i):1095*(i+1)]
    train_sort=np.array(train_sort)
    y=train_sort[7:, -1]

    print(train_sort.shape) # (1095, 10)
    ss=StandardScaler()
    ss.fit(train_sort)
    train_sort=ss.transform(train_sort)

    x=split_x(train_sort, 7)
    x=x[:-2, :]
    y1=y[:-1]
    y2=y[1:]

    x_train, x_val, y1_train, y1_val, y2_train, y2_val=train_test_split(
                                                        x, y1, y2, train_size=0.8, random_state=23)
    
    compile(0, x_train, y1_train, x_val, y1_val)
    compile(1, x_train, y2_train, x_val, y2_val)

    test=x_test[i, :, :, :]
    test=test.reshape(567, 7)
    test=ss.transform(test)
    test=test.reshape(81, 7, 7)

    day7=predict(0, x_train, y1_train, x_val, y1_val, test)
    day8=predict(1, x_train, y2_train, x_val, y2_val, test)
    b.append(day7)
    c.append(day8)
day_7=pd.concat(b, axis=0)
day_8=pd.concat(c, axis=0)

day_7=day_7.to_numpy()
day_8=day_8.to_numpy()
day_7=day_7.reshape(48, 81, 9)
day_8=day_8.reshape(48, 81, 9)
day_7=np.transpose(day_7, axes=(1,0,2))
day_8=np.transpose(day_8, axes=(1,0,2))
day_7=day_7.reshape(3888, 9)
day_8=day_8.reshape(3888, 9)

sub.loc[sub.id.str.contains('Day7'), 'q_0.1':]=day_7.round(2)
sub.loc[sub.id.str.contains('Day8'), 'q_0.1':]=day_8.round(2)

sub.to_csv('./dacon/please.csv', index=False)

ranges = 336
hours = range(ranges)
sub=sub[ranges:ranges+ranges]

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