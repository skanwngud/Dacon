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
    'c:/data/npy/lotte_test_gr.npy'
)

test = test.reshape(-1, 128, 128, 1)

pred = model.predict(
    test
)

submission['prediction'] = np.argmax(pred, axis=-1)
submission.to_csv(
    'c:/data/csv/lotte_2.csv', index = False
)

print('done')

#