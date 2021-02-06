# 필요 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Conv1D, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# csv 파일 불러오기
df_train=pd.read_csv('./dacon/train/train.csv')
sub=pd.read_csv('./dacon/sample_submission.csv')

df_test=list()

for i in range(81):
    temp=pd.read_csv('./dacon/test/'+str(i)+'.csv')
    df_test.append(temp)

df_test=pd.concat(df_test)

# print(df_train.shape) # (52560, 9)
# print(df_test.shape) # (27216, 9)

# 타겟1,2 컬럼값 생성
df_train['Target1']=df_train.iloc[:, -1].shift(-48).fillna(method='ffill')
df_train['Target2']=df_train.iloc[:, -1].shift(-96).fillna(method='ffill')

df_train=df_train.drop(['Day', 'Hour', 'Minute'], axis=1) # 불필요한 컬럼 제거
df_test=df_test.drop(['Day', 'Hour', 'Minute'], axis=1)

print(df_train.info()) # (52560, 8)
print(df_test.info()) # (27216, 6)

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

test=df_test

df_train=df_train.reshape(-1, 48, 8)

def split_x(data, x_row, y_col):
    x=list()
    y=list()
    for i in range(len(data)):
        x_end_number=i+x_row
        y_end_number=x_row+y_col
        if x_end_number>len(data):
            break
        x_temp=data[i:x_end_number, :]
        y_temp=data[x_end_number:y_end_number, -2:]
        x.append(x_temp)
        y.append(y_temp)
    return np.array(x), np.array(y)

x,y=split_x(df_train, 7, 2)

x=x[:, :, :, :-2]

print(x[0].shape)
print(x[0])

print(x.shape) # (1089, 7, 48, 6)
print(y.shape) # (1089, )

# 데이터 전처리
x=x.reshape(-1, 7*48*6)
y=y.reshape(-1, 1)
y1=y[:, 0]
y2=y[:, -1]

print(y[0])

ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)

x=x.reshape(-1, 7, 48, 6)
y1=y1.reshape(-1, 1)
y2=y2.reshape(-1, 1)

x_train, x_test, y_train_1, y_test_1=train_test_split(x, y1, train_size=0.8, random_state=23)
x_train, x_test, y_train_2, y_test_2=train_test_split(x, y2, train_size=0.8, random_state=23)

print(type(x_train))
print(type(y_train_1))
print(type(y_train_2))

x_train=x_train.reshape(-1, 7*48, 6)
x_test=x_test.reshape(-1, 7*48, 6)

# 퀀타일로스 정의
quantile=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss(q, y, pred):
    err=(y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

# 콜백 정의
es=EarlyStopping(monitor='loss', patience=20, mode='auto')
rl=ReduceLROnPlateau(monitor='loss', factor=0.1)

# 모델링
def models():
    model=Sequential()
    model.add(Conv1D(64, 2, padding='same', activation='relu', input_shape=(7*48, 6)))
    model.add(Conv1D(64, 2, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.summary()
    return model

# def models2():
#     model=Sequential()
#     model.add(LSTM(64, activation='relu', input_shape=(48,6)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1))
#     return model

# def models3():
#     model=Sequential()
#     model.add(GRU(64, activation='relu', input_shape=(48, 6)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1))
#     return model


# 컴파일, 훈련
x=list()
for i in quantile:
    model=models()
    model.compile(loss=lambda y_true, y_pred:quantile_loss(i, y_true, y_pred),
                    optimizer='adam')
    model.fit(x_train, y_train_1, validation_split=0.2,
                epochs=200, batch_size=128, callbacks=[es, rl])
    pred=pd.DataFrame(model.predict(test).round(2))
    x.append(pred)
y_pred_1=pd.concat(x, axis=1)
y_pred_1[y_pred_1<0]=0
num_y_pred_1=y_pred_1.to_numpy()

x=list()
for i in quantile:
    model=models()
    model.compile(loss=lambda y_true, y_pred:quantile_loss(i, y_true, y_pred),
                    optimizer='adam')
    model.fit(x_train, y_train_2, validation_split=0.2,
                epochs=200, batch_size=128, callbacks=[es, rl])
    pred=pd.DataFrame(model.predict(test).round(2))
    x.append(pred)
y_pred_2=pd.concat(x, axis=1)
y_pred_2[y_pred_2<0]=0
num_y_pred_2=y_pred_2.to_numpy()

sub.loc[sub.id.str.contains('Day7'), 'q_0.1':]=num_y_pred_1
sub.loc[sub.id.str.contains('Day8'), 'q_0.1':]=num_y_pred_2

sub.to_csv('./dacon/sub_self_conv1d.csv', index=False)

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
