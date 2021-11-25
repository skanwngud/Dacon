import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr

path = "D:/Data/open/"
list_name = "stock_list.csv"

stock_list = pd.read_csv(os.path.join(path, list_name))

stock_name = stock_list['종목명']
stock_code = stock_list['종목코드'].apply(lambda x:str(x).zfill(6))

stock_dict = {name : value for name, value in zip(stock_name, stock_code)}

print(stock_dict)
print(list(stock_dict.values()))

str_date = '20200101'
end_date = '20211101'

# for code in list(stock_dict.values()):
    # stock = fdr.DataReader(str(code), str_date, end_date)
for code in stock_code:
    stock = fdr.DataReader(str(code), str_date, end_date)
    print(stock)
    print(stock.shape)
    print('next')


# print(fdr.DataReader('005930', str_date, end_date))