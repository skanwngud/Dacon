import numpy as np
import pandas as pd

csv = list()
df = pd.DataFrame()
for i in range(5):
    temp = pd.read_csv(
        'c:/data/csv/lotte_{i}.csv'
    )
    