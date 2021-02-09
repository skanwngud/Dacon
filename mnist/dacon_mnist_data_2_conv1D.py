import numpy as np
import pandas as pd

train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
pred=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, \
    Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

datagen=ImageDataGenerator(width_shift_range=(-1, 1), height_shift_range=(-1, 1))
datagen2=ImageDataGenerator()

# print(train.info()) # (2048, 786)
# print(pred.info()) # (20480, 784)

# data
x=train.iloc[:, 2:]
y=train.iloc[:, 0]
pred=pred.iloc[:, 1:]

x=x.to_numpy()
y=y.to_numpy()
pred=pred.to_numpy()

x=x.reshape(-1, 28, 28, 1)
pred=pred.reshape(-1, 28, 28, 1)

# print(x.shape) # (2048, 785)
# print(y.shape) # (2048, )
# print(pred.shape) # (2048, 785)

kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

# model
# model=XGBClassifier(n_estimators=1000, n_jobs=8, learning_rate=0.01)

# model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
#             eval_metric=['mlogloss'], early_stopping_rounds=30, verbose=1)

# score=model.score(x_test, y_test)
for train_index, test_index in kf.split(x,y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

    trainset=datagen.flow(x_train, y_train, batch_size=64)
    testset=datagen2.flow(x_test, y_test)
    predset=datagen2.flow(pred, shuffle=False)
# print('score : ', score) # score :  0.2658536585365854

    trainset=np.squeeze(trainset)
    # testset=testset.reshape(-1, 784, 1)
    predset=np.squeeze(predset)
    # trainset=np.expand_dims(trainset)
    # predset=np.expand_dims(predset)

    model=Sequential()
    model.add(Conv1D(32, 2, padding='same',input_shape=(28, 28)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(32, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, padding='same'))
    # model.add(Conv1D(512, 2, padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(2, padding='same'))
    # model.add(Conv1D(512, 2, padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(2, padding='same'))
    # model.add(Conv1D(256, 2, padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(2, padding='same'))
    # model.add(Conv1D(256, 2, padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Conv1D(32, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dense(1024))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Flatten())
    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    es=EarlyStopping(patience=50)
    rl=ReduceLROnPlateau(patience=10, verbose=1)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
    # model.fit(x_train, y_train, validation_data=(x_test, y_test),
    #             epochs=1000, batch_size=64, callbacks=[es, rl])
    model.fit_generator(trainset, validation_data=(testset),
                        epochs=500, callbacks=[es, rl])
    # fitting
    # model=KerasClassifier(build_fn=model, epochs=100, batch_size=64)
    # cross_val_score(model, x_test, y_test, cv=kf)

    # eval, pred
    loss=model.evaluate_generator(testset)
    # y_pred=model.predict(pred)

print(loss) # [7.086297988891602, 0.22439023852348328]

sub['digit']=np.argmax(model.predict_generator(predset), axis=1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples3.csv', index=False)