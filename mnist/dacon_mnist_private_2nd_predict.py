import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import cv2

import gc
from keras import backend as bek

test = pd.read_csv('../data/dacon/data/test.csv')

x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
x_test = np.where(x_test>=145,255.,x_test)
x_test = x_test/255
x_test = x_test.astype('float32')

test_224=np.zeros([20480,50,50,3],dtype=np.float32)


for i, s in enumerate(x_test):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(50,50),interpolation = cv2.INTER_CUBIC)
    del converted
    test_224[i] = resized
    del resized

bek.clear_session()
gc.collect()






effnet = tf.keras.applications.EfficientNetB3(
    include_top=True,
    weights=None,
    input_shape=(50,50,3),
    classes=10,
    classifier_activation="softmax",
)



loaded_model = Sequential()
loaded_model.add(effnet)


loaded_model.compile(loss="categorical_crossentropy",
            optimizer=RMSprop(lr=2e-3),
            metrics=['accuracy'])

del x_test
del test
results = np.zeros( (20480,10),dtype=np.float16)

for j in range(50):
  filepath_val_acc="../data/modelcheckpoint/model/effi_model_aug"+str(j+1)+".ckpt"
  loaded_model.load_weights(filepath_val_acc)
  results = results + loaded_model.predict(test_224)
  
  del filepath_val_acc
  bek.clear_session()
  gc.collect()
  
np.savetxt('../data/dacon/data/results.csv',results ,delimiter=',')  ## 유사도 판정표




# Predict 결과를 앙상블하여 최종적인 예측값 결정


submission = pd.read_csv('../data/dacon/data/sample_submission.csv')
submission['digit'] = np.argmax(results, axis=1)
# model.predict(x_test)
submission.head()
submission.to_csv('../data/dacon/data/loadtest2.csv', index=False)