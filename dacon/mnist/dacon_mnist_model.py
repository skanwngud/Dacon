import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

datagen=ImageDataGenerator(width_shift_range=(-1, 1), height_shift_range=(-1, 1))
datagen2=ImageDataGenerator()

kf=StratifiedKFold(n_splits=50, shuffle=True, random_state=22)

train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
pred=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

x=train.iloc[:, 2:] # Letter 제외
y=train.iloc[:, 0] # Digit 값
pred=pred.iloc[:, 1:] # Letter 제외

# numpy 전환
x=x.to_numpy()
y=y.to_numpy()
pred=pred.to_numpy()

print(x.shape) # (2048, 784)
print(y.shape) # (2048, )

x=x.reshape(-1, 28, 28, 1)/255.
pred=pred.reshape(-1, 28, 28, 1)/255.

# y=to_categorical(y)


for train_index, validation_index in kf.split(x, y):
    x_train=x[train_index]
    x_val=x[validation_index]
    y_train=y[train_index]
    y_val=y[validation_index]

    trainset=datagen.flow(x_train, y_train, batch_size=64)
    testset=datagen2.flow(x_val, y_val)
    predset=datagen2.flow(pred)

    model=load_model('../data/modelcheckpoint/weight_6_0.9512_0.2501.hdf5')

sub['digit']=np.argmax(model.predict_generator(predset), axis=1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples_3.csv', index=False)