# 5개 csv 파일 저장
# 5개 중 가장 많이 나온 값 저장

import numpy as np
import pandas as pd
import glob
import datetime

from scipy import stats

str_time = datetime.datetime.now()

csv = list()
all_file = glob.glob(
    'c:/data/csv/lotte_*.csv'
)

print('1', datetime.datetime.now() - str_time)

for filename in all_file:
    temp = pd.read_csv(filename, index_col=0, header=0)
    csv.append(temp)

print('2', datetime.datetime.now() - str_time)

all_csv = pd.concat(csv, axis=1, ignore_index=True)
# all_csv.to_csv(
#     'c:/data/csv/lotte_all.csv', index = False
# )

print('3', datetime.datetime.now() - str_time)

# pred = pd.read_csv(
#     'c:/data/csv/lotte_all.csv', header=0
# )

# print('4', datetime.datetime.now() - str_time)

# a = pred.iloc[1, :]

# print(pred.info())
# print(pred.head())
# print(type(pred))
# print(pred.iloc[1, :])

# pred = np.array(pred)
pred = np.array(all_csv)

print(pred.shape) # (72000, 5)
print(type(pred))
print(pred[1, 1]) # 208

print(np.unique(pred[:5, :]))

# for i in range(72000):
#     temp = pred[i, :]

aa = stats.mode(pred, axis = 1).mode

print(aa)
print(aa.shape)

b = pd.DataFrame(aa)

b.to_csv('c:/data/csv/lotte_democracy.csv', index=False)
# count = stats.mode((72220,5), axis = 1).count

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

submission['prediction'] = b.iloc[:, 0]

submission.to_csv('c:/data/csv/lotte_submission_all.csv', index = False)