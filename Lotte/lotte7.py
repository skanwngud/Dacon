# 파일 중 최다값만 모아 출력함

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

# 파일 로드 후 리스트에 저장한 뒤 넘파이로 변환
x = []
for i in range(5):
    df = pd.read_csv(f'c:/LPD_competition/lotte_{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)
print(x.shape)  # (5, 72000, 1)
print(x[0])

df = pd.read_csv(f'c:/LPD_competition/lotte_{i}.csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        a = []
        for k in range(5):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')  # 5개 파일의 중앙값
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('c:/LPD_competition/lotte_final.csv') 