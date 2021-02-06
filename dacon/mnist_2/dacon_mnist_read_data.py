import numpy as np
import pandas as pd

train=pd.read_csv('../data/dacon/data2/imagsets.csv')

print(train.info())
print(train.head())