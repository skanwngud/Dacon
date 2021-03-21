# model predict
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

model = load_model(
    'c:/data/modelcheckpoint/lotte.hdf5'
)

test = np.load(
    'c:/data/npy/lotte_test.npy'
)

# test = test.reshape(-1, 128, 128, 3)

pred = model.predict(
    test
)

submission['prediction'] = np.argmax(pred, axis=-1)
submission.to_csv(
    'c:/data/csv/lotte__2.csv', index = False
)

print('done')

# val_acc 0.87 : 45
# val_acc 0.98 : 55
# val_acc 0.97 : 