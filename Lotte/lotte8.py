# 5개 csv 파일 저장
# 5개 중 가장 많이 나온 값 저장

import numpy as np
import pandas as pd
import glob
from scipy import stats

csv = list()
all_file = glob.glob(
    'c:/data/csv/best_lotte_*.csv'
)

for filename in all_file:
    temp = pd.read_csv(filename, index_col=0, header=0)
    csv.append(temp)

all_csv = pd.concat(csv, axis=1, ignore_index=True)
all_csv.to_csv(
    'c:/data/csv/lotte_all_csv.csv', index = False
)

pred = pd.read_csv(
    'c:/data/csv/lotte_all_csv.csv', header=0
)

a = pred.iloc[1, :]

print(pred.info())
print(pred.head())
print(type(pred))
print(pred.iloc[1, :])

pred = np.array(pred)

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

b.to_csv('c:/data/csv/lotte_b.csv', index=False)
# count = stats.mode((72220,5), axis = 1).count

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

submission['prediction'] = b.iloc[:, 0]

submission.to_csv('c:/data/csv/lotte_submiss ion.csv', index = False)