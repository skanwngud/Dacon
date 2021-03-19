# 주말용 전이학습 모델 돌리기용

import numpy as np
import pandas as pd

from tensorflow.keras.applications import MobileNet, DenseNet201, EfficientNetB4,\
    InceptionResNetV2, ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten,\
    BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = np.load(
    'c:/data/npy/lotte_x.npy'
)

y = np.load(
    'c:/data/npy/lotte_y.npy'
)

test = np.load(
    'c:/data/npy/lotte_test.npy'
)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

batch_size = 32
epochs = len(x)//batch_size

es = EarlyStopping(
    patience=50,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=20,
    verbose=1
)

datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True,
    width_shift_range=(-1, 1),
    height_shift_range=(-1, 1)
)

datagen2 = ImageDataGenerator()

x = x.reshape(-1, 128, 128, 3)
test = test.reshape(-1, 128, 128, 3)


x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    train_size = 0.8,
    random_state = 23
)

x_train_set = datagen.flow(
    x_train,
    batch_size = batch_size
)

x_val_set = datagen.flow(
    x_val,
    batch_size = batch_size
)

test_set = datagen2.flow(
    test,
    batch_size = batch_size
)

model_list = [MobileNet, EfficientNetB4, ResNet152, DenseNet201, InceptionResNetV2]
# efficientnetB4 가 가장 점수가 높게 나왔음

count = 0
for i in model_list:
    print(i)
    models = i(
        include_top = False,
        input_shape = (128, 128, 3)
    )
    models.trainable = True

    model = Sequential()
    model.add(models)
    model.add(Conv2D(128, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('swish')) # 깊은 신경망에서는 relu 보다 swish 가 더 효과적이라는 논문 결과가 있다.
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Conv2D(256, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('swish'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1000, activation='softmax'))

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics='acc'
    )

    mc = ModelCheckpoint(
        'c:/data/modelcheckpoint/lotte_' + str(count) + '.hdf5',
        save_best_only = True,
        verbose = 1
    )

    model.fit(
        x_train, y_train,
        validation_data = (x_val, y_val),
        epochs = 500,
        batch_size = batch_size,
        callbacks = [es, rl, mc]
    )

    model.load_weights(
        'c:/data/modelcheckpoint/lotte_' + str(count) + '.hdf5'
    )

    pred = model.predict(
        test
    )

    results = model.predict(test)

    results += results/5

    submission['prediction'] = np.argmax(pred, axis = -1)
    submission.to_csv(
        'c:/data/csv/lotte_' + str(count) + '.csv',
        index = False
    )

    count += 1

    print(str(i) + ' done')

submission['prediction'] = np.argmax(results, axis = -1)
submission.to_csv(
    'c:/data/csv/lotte_average.csv',
    index = False
)