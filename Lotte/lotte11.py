# 민주주의 최종

import pandas as pd
import numpy as np
import datetime
import glob

from scipy import stats

csv = list()

for i in range(14):
    temp = pd.read_csv(
        'c:/data/csv/best_lotte_%s.csv'%i
    )
    temp2 = temp['prediction']
    csv.append(temp2)

all_csv = pd.concat(csv, axis=1)

print(type(all_csv))
print(all_csv.info())

demo = stats.mode(all_csv, axis=1).mode

print(demo)
print(type(demo))

democracy = pd.DataFrame(demo)

democracy.to_csv(
    'c:/data/csv/final_democracy.csv',
    index = False
)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

submission['prediction'] = democracy.iloc[:, 0]
submission.to_csv(
    'c:/data/csv/real_final_democracy.csv', index = False
)