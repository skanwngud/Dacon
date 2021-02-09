'''
# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow

from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator # 이미지 생성자
from sklearn import metrics

from sklearn.model_selection import train_test_split

import cv2 # openCV

import gc
# garbage collection : 현대 언어에서 쓰이는 메모리 처리 과정
from keras import backend as bek

# data
train=pd.read_csv('../data/dacon/data/train.csv')
test=pd.read_csv('../data/dacon/data/test.csv')

# make train data set
x_train=train.drop(['id', 'letter', 'digit'], axis=1).values
x_train=x_train.reshape(-1, 28, 28, 1)

x_train=np.where((x_train<=20)&(x_train!=0), 0., x_train)

x_train=x_train/255
x_train=x_train.astype('float32')

# make test data set (only digit)
y=train['digit']
y_train=np.zeros((len(y), len(y.unique()))) # 0 ~ 9 까지의 10행을 추가한다

for i, digit in enumerate(y): # y 안의 i 와 digit
    y_train[i, digit]=1 # 일반적인 OneHotEncoding 방법

train_224=np.zeros([2048, 50, 50, 3], dtype=np.float32) # 50, 50 사이즈의 검은 배경을 만든다

for i,s in enumerate(x_train):
    converted=cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    # 기존 데이터를 흑백에서 컬러로 바꾸어 특성을 강조시킨다
    resized=cv2.resize(converted, (50, 50), interpolation=cv2.INTER_CUBIC)
    # 28, 28 사이즈의 기존 데이터를 정사각형 형태의 50, 50 사이즈로 늘린다
    # cv2.INTER_LINEAR 도 있는데 이건 정사각형이 아닌 선형 형태로 상하좌우로만 늘림
    del converted # 변수초기화
    train_224[i]=resized
    del resized
    bek.clear_session()
    gc.collect()

# 훈련 데이터 증폭을 위해 데이터 생성
datagen=ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range=10,
        validation_split=0.2
)

# 검증 데이터는 default 값으로 한다
valgen=ImageDataGenerator()

from keras.callbacks import LearningRateScheduler, EarlyStopping

# EfficientNet 모델 사용
# EfficientNet 모델은 기존 CNN 모델에서의 단점을 보완한 모델로써
# 채널을 늘리거나 레이어를 깊게 쌓거나 노드 수를 늘리는 것과 같은 3가지의 방법으로
# 성능을 개선하는 기존 모델보다 훨씬 빠르고 성능이 높게 나온다
def create_model():
    effnet=tf.keras.applications.EfficientNetB3(
        include_top=True,
        weights=None,
        input_shape=(50, 50, 3),
        classes=10,
        classifier_activation='softmax',
    )

    model=Sequential()
    model.add(effnet)
    
    model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(lr=initial_learningrate),
                    metrics=['acc'])
    return model

# optimizer learningrate
initial_learningrate=2e-3

from sklearn.model_selection import RepeatedKFold, KFold

kf=RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores=[]
Fold=1
results=np.zeros((20480, 10))

def lr_decay(epoch):
    return initial_learningrate*0.99**epoch

x_test=test.drop(['id', 'letter'], axis=1).values
x_test=x_test.reshape(-1, 28, 28, 1)
x_test=np.where((x_test<=20)&(x_test!=0), 0., x_test)
x_test=x_test/255
x_test=x_test.astype('float32')

test_224=np.zeros([20480, 50, 50, 3], dtype=np.float32)

for i, s in enumerate(x_test):
    converted=cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized=cv2.resize(converted, (50, 50), interpolation=cv2.INTER_CUBIC)
    del converted
    test_224[i]=resized
    del resized

bek.clear_session()
gc.collect()

results=np.zeros((20480, 10), dtype=np.float32)

for train, val in kf.split(train_224):
    initial_learningrate=2e-3
    es=EarlyStopping(patience=50)
    filepath_val_acc='../data/modelcheckpoint/effi_model_aug'+str(Fold)+'.ckpt'
    checkpoint_val_acc=ModelCheckpoint(filepath_val_acc, monitor='val_acc',
                    verbose=1, save_best_only=True, save_weights_only=True)
    
    gc.collect()
    bek.clear_session()

    print('Fold : ', Fold)
    
    x_train=train_224[train]
    x_val=train_224[val]
    x_train=x_train.astype('float32')
    x_val=x_val.astype('float32')

    y_train=y_train[train]
    y_val=y_train[val]

    model=create_model()

    training_generator=datagen.flow(x_train, y_train, batch_size=4, seed=7, shuffle=True)
    validation_generator=valgen.flow(x_val, y_val, batch_size=4, seed=7, shuffle=True)
    model.fit(training_generator, epochs=150, callbacks=[LearningRateScheduler(lr_decay), es, checkpoint_val_acc],
                shuffle=True,
                validation_data=validation_generator,
                steps_per_epoch=len(x_train)/32)

    del x_train
    del x_val
    del y_train
    del y_val

    gc.collect()
    bek.clear_session()
    model.load_weights(filepath_val_acc)
    results=results+model.predict(test_224)

    Fold=Fold+1

submission=pd.read_csv('../data/dacon/data/sample_submission.csv')
submission['digit']=np.argmax(results, axis=1)

submission.head()
submission.to_csv('../data/dacon/data/private2.csv', index=False)
'''
# private 2등 코드
# 300, 300 >> 50, 50

# warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
# from keras.utils import np_utils
import cv2

import gc
from keras import backend as bek

########################################
train = pd.read_csv('../data/dacon/data/train.csv')
print(train.shape)  # (2048, 787)

submission = pd.read_csv('../data/dacon/data/sample_submission.csv')
print(submission.shape) # (20480, 2)

test = pd.read_csv('../data/dacon/data/test.csv')
print(test.shape)   # (20480, 786)
########################################

from sklearn.model_selection import train_test_split

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)

x_train = np.where((x_train<=20)&(x_train!=0) ,0.,x_train)  # 특징이 낮은 것들은 모두 0으로 반환

x_train = x_train/255
x_train = x_train.astype('float32')

y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y):
    y_train[i, digit] = 1

train_224=np.zeros([2048,50,50,3],dtype=np.float32) # [2048,50,50,3] 의 검은색 배경을 만들어 준다.

for i, s in enumerate(x_train):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) # 컬러 이미지로 바꿔줌
    resized = cv2.resize(converted,(50,50),interpolation = cv2.INTER_CUBIC) # 50, 50으로 리사이즈
    del converted
    train_224[i] = resized
    del resized
    bek.clear_session()
    gc.collect()

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

datagen = ImageDataGenerator(   # 이미지 증폭
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2)

valgen = ImageDataGenerator()

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping

def create_model() :
    
    effnet = tf.keras.applications.EfficientNetB3(
        include_top=True,
        weights=None,
        input_shape=(50,50,3),
        classes=10,
        classifier_activation="softmax",
    )
    model = Sequential()
    model.add(effnet)


    model.compile(loss="categorical_crossentropy",
                optimizer=RMSprop(lr=initial_learningrate),
                metrics=['accuracy'])
    return model

initial_learningrate=2e-3  

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores = []
Fold = 1
results = np.zeros((20480,10))

def lr_decay(epoch):#lrv
    return initial_learningrate * 0.99 ** epoch


x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
# x_test = np.where(x_test>=145,255.,x_test)
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



results = np.zeros( (20480,10),dtype=np.float32)


for train, val in kfold.split(train_224): 
    # if Fold<25:
    #   Fold+=1
    #   continue
    initial_learningrate=2e-3  
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=50)      
    filepath_val_acc="../data/modelcheckpoint/effi_model_aug"+str(Fold)+".ckpt"
    checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)


    gc.collect()
    bek.clear_session()
    print ('Fold: ',Fold)
    
    X_train = train_224[train]
    X_val = train_224[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = y_train[train]
    Y_val = y_train[val]

    model = create_model()


    training_generator = datagen.flow(X_train, Y_train, batch_size=16,seed=7,shuffle=True)
    validation_generator = valgen.flow(X_val, Y_val, batch_size=16,seed=7,shuffle=True)
    model.fit(training_generator,epochs=200,callbacks=[LearningRateScheduler(lr_decay),es,checkpoint_val_acc],
               shuffle=True,
               validation_data=validation_generator,
               steps_per_epoch =len(X_train)//16
               )

    del X_train
    del X_val
    del Y_train
    del Y_val

    gc.collect()
    bek.clear_session()
    model.load_weights(filepath_val_acc)
    results = results + model.predict(test_224)
    
    Fold = Fold +1

submission['digit'] = np.argmax(results, axis=1)
# model.predict(x_test)
submission.head()
submission.to_csv('../data/dacon/data/0204_7_private2_sub.csv', index=False)