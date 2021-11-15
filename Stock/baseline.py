import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr

from sklearn.linear_model import LinearRegression
from tqdm import tqdm

path = "D:/Data/open"
list_name = "stock_list.csv"
sample_name = 'sample_submission.csv'
sample_submission = pd.read_csv(os.path.join(path, sample_name))

stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x:str(x).zfill(6))

print(stock_list) # (370, 3)

start_date = "20100101"
end_date = "20211105"

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime("%V")
Business_days = pd.DataFrame(pd.date_range(start_date,end_date, freq='B'), columns=['Date'])

print(f"WeekDay of 'Start_date' : {start_weekday}") # 0
print(f"Num of Weeks to 'end_date' : {max_weeknum}") # 44
print(f"How Many 'Business Days' : {Business_days.shape}") # (220, 1)

print(Business_days.head())

# Baseline Model
# 이번주 월~금요일의 패턴을 학습해 다음주 월~금요일의 패턴을 예측하는 모델
# 이 과정을 모든 컬럼 (370개) 에 적용

sample_code = stock_list.loc[0, '종목코드']

model = LinearRegression()

for code in tqdm(stock_list['종목코드'].values):
    sample = fdr.DataReader(sample_code, start=start_date, end=end_date)[['Close']].reset_index()
    sample = pd.merge(Business_days, sample, how='outer')
    sample['weekday'] = sample.Date.apply(lambda x : x.weekday())
    sample['weeknum'] = sample.Date.apply(lambda x : x.strftime('%V'))
    sample.Close = sample.Close.ffill()
    sample = pd.pivot_table(data=sample, values='Close', columns='weekday', index='weeknum')

    # print(sample.head())


    x = sample.iloc[0:-2].to_numpy() # 첫 번째 종목에 대한 가격
    # print(x.shape) # (42, 5)

    y = sample.iloc[1:-1].to_numpy()
    y_0 = y[:,0]
    y_1 = y[:,1]
    y_2 = y[:,2]
    y_3 = y[:,3]
    y_4 = y[:,4]
    y_values = [y_0, y_1, y_2, y_3, y_4]
    # print(y_values)

    x_public = sample.iloc[-2].to_numpy() # 2021년 11월 1일부터 2021년 11월 5일까지의 데이터 예측
    # print(x_public)

    predictions = []
    for y_value in y_values:
        model.fit(x,y_value)
        prediction = model.predict(np.expand_dims(x_public,0))
        predictions.append(prediction[0])

    sample_submission.loc[:,code] = predictions * 2

sample_submission.isna().sum().sum()
sample_submission.columns

columns = list(sample_submission.columns[1:])
columns = ['Day'] + [str(x).zfill(6) for x in columns]

sample_submission.columns = columns

sample_submission.to_csv('Baseline_Linear.csv', index=False)
