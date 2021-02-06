# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

train=pd.read_csv('./dacon/train/train.csv') # import train data
submission=pd.read_csv('./dacon/sample_submission.csv') # import submission data

def preprocess_data(data, is_train=True):
    temp=data.copy()
    temp=temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    if is_train==True: # add to Target1,2
        temp['Target1']=temp['TARGET'].shift(-48).fillna(method='ffill') # Day7
        temp['Target2']=temp['TARGET'].shift(-96).fillna(method='ffill') # Day8
        return temp.iloc[:-96]
    elif is_train==False: # test data
        temp=temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :]

df_train=preprocess_data(train)

df_test=list()

for i in range(81):
    file_path='./dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path)
    temp=preprocess_data(temp, is_train=False) # test data, no add to Target1,2
    df_test.append(temp)

x_test=pd.concat(df_test) # merge to files

# train test set split
x_train_1, x_val_1, y_train_1, y_val_1=train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], train_size=0.7, random_state=32)
x_train_2, x_val_2, y_train_2, y_val_2=train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], train_size=0.7, random_state=32)
# df_train.iloc[:, :-2] except Target1,2 columns
# df_train.iloc[:, -2] Day7
# df_train.iloc[:, -1] Day8

quantile=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# define model
def LGBM(q, x_train, y_train, x_val, y_val, x_test):
    model=LGBMRegressor(objective='quantile', alpha=q,
                        n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)
                        # n_estimators = epochs
    model.fit(x_train, y_train, eval_metric=['qunatile'],
                eval_set=[(x_val, y_val)], early_stopping_rounds=300, verbose=500)
                # verbos=500 : 500 마다 표시
    pred=pd.Series(model.predict(x_test).round(2))
    return pred, model

# define predict model
def train_data(x_train, y_train, x_val, y_val, x_test):
    LGBM_models=list()
    LGBM_actual_pred=pd.DataFrame()
    
    for q in quantile:
        print(q)
        pred, model=LGBM(q, x_train, y_train, x_val, y_val, x_test)
        LGBM_models.append(model)
        LGBM_actual_pred=pd.concat([LGBM_actual_pred, pred], axis=1)

    LGBM_actual_pred.columns=quantile
    return LGBM_models, LGBM_actual_pred

# train Day7
models_1, results_1=train_data(x_train_1, y_train_1, x_val_1, y_val_1, x_test)

# train Day8
models_2, results_2=train_data(x_train_2, y_train_2, x_val_2, y_val_2, x_test)

submission.loc[submission.id.str.contains('Day7'), 'q_0.1':]=results_1.sort_index().values
submission.loc[submission.id.str.contains('Day8'), 'q_0.1':]=results_2.sort_index().values

submission.to_csv('./dacon/baseline_v1.csv', index=False)

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