import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr

path = "D:/Data/open/"
list_name = "stock_list.csv"
sample_name = "sameple_submission.csv"

stock_list = pd.read_csv(os.path.join(path, list_name))

stock_name = stock_list['종목명']
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x:str(x).zfill(6))

# stock_dict = {name : value for name, value in zip(stock_name, stock_code)}

# print(stock_dict)
# print(list(stock_dict.values()))

str_date = '20200101'
end_date = '20211101'

str_weekday = pd.to_datetime(str_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime("%V")
Business_days = pd.DataFrame(pd.date_range(str_date, end_date, freq='B'), columns=['Date'])

print(Business_days)
#
# sample_code = stock_list[0, '종목코드']


stock = fdr.DataReader('005930', str_date, end_date)
stock = pd.merge(Business_days, stock, how='outer')
stock['weekday'] = stock.Date.apply(lambda x:x.weekday())
stock['weeknum'] = stock.Date.apply(lambda x:x.strftime('%V'))
stock.Close = stock.Close.ffill()
stock = pd.pivot_table(data=stock, values='Close', columns='weekday', index='weeknum')

print(stock)
print(stock.head())
print(stock.describe())
print(stock.isnull().sum())

print(stock.iloc[:4, :])
print(stock.iloc[:3,:])
