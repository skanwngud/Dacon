import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train=pd.read_csv('./dacon/train/train.csv', header=0, index_col=0)
sub=pd.read_csv('./dacon/sample_submission.csv', header=0, index_col=0)

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

df_train=preprocess_data(x_train)

df_test=list()

for i in range(81):
    file_path = './dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path)
    df_test.append(temp)

df_test = pd.concat(df_test)

# df_train.to_csv('./dacon/train/train_1.csv')
# df_test.to_csv('./dacon/test/test_1.csv')

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

print('df_train.shape:',df_train.shape) # (52464, 10)
print('df_test.shape:',df_test.shape) # (27216, 9)

df_train=df_train[:, :-2]
print('df_train.shape:', df_train.shape) # (52464, 8)


def split_x(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y=split_x(df_train, 7, 2)

print('x.shape:',x.shape) # (1087, 7, 48, 8)
print('y.shape:',y.shape) # (1087, 2, 48, 8)
print('df_test.shape:',df_test.shape) # (81, 7, 48, 6)

