import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)

x=train.iloc[:, 2:]

pca=PCA()
x=pca.fit_transform(x)

cumsum=np.cumsum(pca.explained_variance_ratio_)

print(np.argmax(cumsum>=0.99)+1) # 277
# 이미지 데이터에서 pca 를 사용하면 column 이 삭제가 되면서 특성이 사라질 수도 있기 때문에,
# 사용할 때에는 유의를 해야한다. (실제로 결과값이 더 안 좋게 나옴)