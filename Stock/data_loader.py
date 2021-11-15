import pandas as pd
import os
import FinanceDataReader as fdr

path = "D:/Data/open/"
list_name = "Stock_List.csv"
stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x:str(x).zfill(6))

print(stock_list)

start_date = "20200101"
end_date = "20201231"
sample_code = "005930"
stock = fdr.DataReader(sample_code, start=start_date, end=end_date)
print(stock)