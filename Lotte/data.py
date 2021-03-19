import tensorflow
import numpy as np
import glob
import cv2

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization,\
    Activation, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB7

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from PIL import Image

datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True,
    rescale=1./255,
    height_shift_range=(-1, 1),
    width_shift_range=(-1, 1)
)

datagen2 = ImageDataGenerator(
    rescale=1./255
)

# xy_data = datagen.flow_from_directory(
#     directory='c:/LPD_competition/train/',
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='categorical'
# )

# np.save('c:/LPD_competition/x_train.npy', arr = xy_data[0][0])
# np.save('c:/LPD_competition/y_train.npy', arr = xy_data[0][1])

# x_train = np.load('c:/LPD_competition/x_train.npy')
# y_train = np.load('c:/LPD_competition/y_train.npy')

pred = list()
for i in range(72000):
    img = cv2.imread(
        'c:/LPD_competition/test_1/test/%s.jpg'%i
    )
    img = cv2.resize(img, (128, 128))
    img = np.array(img)
    pred.append(img)

pred = np.array(pred)

np.save('c:/LPD_competition/pred.npy', arr = pred)

print(pred.shape)
