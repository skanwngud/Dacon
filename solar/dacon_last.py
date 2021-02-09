import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, LSTM, Conv1D, Conv2D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train=pd.read_csv('./dacon/train/train.csv', index_col=None, header=0)
submission=pd.read_csv('./dacon/sample_submission.csv')

df_train=train.iloc[:,[1,3,4,5,6,7,8]] # Hour, Minute 컬럼 제외

print(df_train.shape) # (52560, 7)

# Hour, Minute 컬럼 제외, 하루치만 가져옴
def preproccess_data(data):
    temp=data.copy() # temp 변수 설정
    return temp.iloc[-48:, [1,3,4,5,6,7,8]]

df_test=list()
for i in range(81):
    file_path='./dacon/test/'+str(i)+'.csv'
    test=pd.read_csv(file_path)
    test=preproccess_data(test)
    df_test.append(test)

df_test=pd.concat(df_test)
df_test.to_csv('./dacon/ss.csv')

print(df_test.shape) # (3888, 7)
print(df_test.iloc[0])

def Add_feature(data):
    c=243.12
    b=17.62
    gemma=(b*(data['T'])/(c+(data['T'])))+np.log(data['RH']/100)
    dp=(c*gemma)/(b-gemma)
    data['cos']=np.cos(np.pi/2-np.abs(data['Hour']%12-6)/6*np.pi/2)
    data.insert(1, 'Td', dp)
    data.insert(1, 'T-Td', data['T']-data['Td'])
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis=1, inplace=True)
    data.drop(['RH'], axis=1, inplace=True)
    data.drop(['Td'], axis=1, inplace=True)
    return data

df_train=Add_feature(df_train)
df_test=Add_feature(df_test)

print(df_train.shape) # (52560, 8)
print(df_test.shape) # (3888, 8)

day7=df_train['TARGET'].shift(-48) # day7 치 타겟값
day8=df_train['TARGET'].shift(-96) # day8 치 타겟값

df_train=pd.concat([df_train, day7, day8], axis=1) # TARGET 값 정렬
df_train=df_train.iloc[:-96, :] # TARGET 값을 옆으로 붙였기 때문에 2일치가 비어있음

print(df_train.shape) # (52464, 10) - Day7,8 column 이 추가 됨

z=df_train.values

def split_x(z, x_row, x_col, y_row, y_col):
    x, y=list(), list()
    for i in range(len(z)):
        if i > len(z)-x_row:
            break
        temp_x=z[i:i+x_row, :x_col]
        temp_y=z[i:i+x_row, x_col:x_col+y_col]
        x.append(temp_x)
        y.append(temp_y)
    return np.array(x), np.array(y)

x, y=split_x(z, 48, 8, 48, 2) # 1일치씩 자르기

print(x.shape) # (52417, 48, 8)
print(y.shape) # (52417, 48, 2)

# DataFrame to numpy
df_test=df_test.to_numpy()
print(df_test.shape) # (48, 7)
df_test=df_test.reshape(int(df_test.shape[0]/48), 48, df_test.shape[1])
df_test=df_test.reshape(df_test.shape[0], df_test.shape[1]*df_test.shape[2])
print(df_test.shape)

# 2차원으로 변환
x=x.reshape(-1, x.shape[1]*x.shape[2])
y=y.reshape(-1, y.shape[1]*y.shape[2])

# train, test dataset 으로 분리
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)
x_train, x_val, y_train, y_val=train_test_split(x_train,y_train, train_size=0.8, random_state=23)

print(x_train.shape) # (33546, 384)
print(test.shape) # (1, 336)
# scaler
ss=StandardScaler()
ss.fit(x_train)
x_train=ss.transform(x_train)
x_test=ss.transform(x_test)
x_val=ss.transform(x_val)
df_test=ss.transform(df_test)

# 변수 설정해서 좀 더 쉽게 코딩하기
num1=8
num2=2

# 2차원 -> 4차원으로 변환
x_train=x_train.reshape(x_train.shape[0], 1, int(x_train.shape[1]/num1), num1)
x_test=x_test.reshape(x_test.shape[0], 1, int(x_test.shape[1]/num1), num1)
x_val=x_val.reshape(x_val.shape[0], 1, int(x_val.shape[1]/num1), num1)
df_test=df_test.reshape(df_test.shape[0], 1, int(df_test.shape[1]/num1), num1)

y_train=y_train.reshape(y_train.shape[0], 1, int(y_train.shape[1]/num2), num2)
y_test=y_test.reshape(y_test.shape[0], 1, int(y_test.shape[1]/num2), num2)
y_val=y_val.reshape(y_val.shape[0], 1, int(y_val.shape[1]/num2), num2)

print(x_train.shape) # (33546, 1, 48, 8)
print(y_train.shape) # (33546, 1, 48, 2)
print(df_test.shape) # (81, 1, 48, 8)

quantile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 퀀타일 로스 정의
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

# callbacks 정의
es=EarlyStopping(monitor='loss', mode='auto', patience=30)
rl=ReduceLROnPlateau(monitor='loss', mode='auto', patience=10, factor=0.1)

# 모델링
def models():
    model=Sequential()
    model.add(Conv2D(64, 2, padding='same', activation='relu', input_shape=(1, 48, 8)))
    model.add(Conv2D(128, 2, padding='same', activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(64, 2, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Reshape((48, 2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))
    # model.summary()

    return model

# 컴파일, 훈련
for q in quantile_list:
    print(str(q)+'번')
    model1=models()
    model1.compile(loss=lambda y_true, y_pred:quantile_loss(q, y_true, y_pred),
                    optimizer='adam', metrics=['mae'])
    hist=model1.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=200, batch_size=256, callbacks=[es, rl])
    
# 평가, 예측
    loss=model1.evaluate(x_test, y_test)
    pred1=model1.predict(df_test)
    print(pred1.shape)
    pred2=pd.DataFrame(pred1.reshape(pred1.shape[0]*pred1.shape[1], pred1.shape[2]))

    y_pred1=pd.concat([pred2], axis=1)
    y_pred1[pred2<0]=0
    y_predict1=y_pred1.to_numpy()

    submission.loc[submission.id.str.contains('Day7'), 'q_'+str(q)]=y_predict1[:, 0].round(2)
    submission.loc[submission.id.str.contains('Day8'), 'q_'+str(q)]=y_predict1[:, 1].round(2)

# csv 파일 생성
submission.to_csv('./dacon/ssss.csv', index=False)

# 시각화
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