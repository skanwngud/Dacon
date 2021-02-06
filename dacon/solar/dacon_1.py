import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 데이터 임포트
df_train=pd.read_csv('./dacon/train/train.csv')
df_sub=pd.read_csv('./dacon/sample_submission.csv')

df_test=list()
for i in range(81):
    test=pd.read_csv('./dacon/test/'+str(i)+'.csv')
    df_test.append(test)

df_test=pd.concat(df_test)

# GHI 정의
def Add_feature(data):
    data['cos']=np.cos(np.pi/2-np.abs(data['Hour']%12-6)/6*np.pi/2)
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis=1, inplace=True)
    return data

# GHI 추가
def preprocess_data(data, is_train=True):
    data=Add_feature(data)
    temp=data.copy()
    temp=temp[['GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET']]
    return temp.iloc[:, :]

# 타겟값 생성 후 제일 뒤로 보내고 열맞춤
    if is_train==True:
        temp['Target1']=temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2']=temp['TARGET'].shift(-96).fillna(method='ffill')
        temp=temp.dropna()
        return temp.iloc[:-96, :] # 7,8 일차를 예측하기 위해 이틀은 뺀다

    elif is_train==False:
        return temp.iloc[-48*5: 1:] # 훈련에 5일치만 사용

print(df_train.info()) # (52560, 9)

# 같은 시간대로 묶기
def same_train(train):
    temp=train.copy() # temp 를 train 으로 카피
    x=list()
    final_x=list()
    for i in range(48):
        same_time=pd.DataFrame()
        for j in range(int(len(temp)/48)):
            temp=temp.iloc[i+48*j, :]
            temp=temp.to_numpy()
            temp=temp.reshape(1, temp.shape[0])
            temp=pd.DataFrame(temp)
            same_time=pd.concat([same_time, temp])
        x=same_time.to_numpy()
        final_x.append(x)
        return np.array(final_x)

# 데이터 스플릿
def split_x(dataset, time_steps):
    x=list()
    y=list()
    for i in range(len(dataset)):
        x_end_number=i+time_steps
        y_end_number=x_end_number-1
        if x_end_number>len(dataset):
            break
        temp_x=dataset[i:x_end_number, 1:-2]
        temp_y=dataset[y_end_number, -2:]
        x.append(temp_x)
        y.append(temp_y)
    return np.array(x), np.array(y)

df_train=preprocess_data(df_train)

same_time=same_train(df_train)

x=list()
y=list()
for i in range(48):
    temp1, temp2=split_x(same_time[i], 5)
    x.append(temp1)
    y.append(temp2)

x=np.array(x)
y=np.array(y)

y=y.reshape(48, -1, 1, 2)

df_tets=preprocess_data(df_train, is_train=False)

x_pred=np.array(df_test)

x_train, x_test, y_train, y_test=train_test_split(x,y,
                                    train_size=0.8, random_state=32)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train,
                                    train_size=0.8, random_state=32)

print(x_train.shape)
print(y_train.shape)