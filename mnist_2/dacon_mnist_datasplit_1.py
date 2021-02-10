import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import PIL

from PIL import Image

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Activation

import datetime

print('데이터 로드 시작')

time=datetime.datetime.now()

images=list()
for i in range(50000):
    filepath='C:\data\dacon\data2\dirty_mnist\%05d.png'%i
    img=Image.open(filepath)
    img=np.array(img)/255
    img=img.astype('float32')
    images.append(img)

np.save('C:/data/dacon/data2/train_set.npy', arr=images)

images_num=list()
for i in range(5000):
    filepath='C:/data/dacon/data2/test_dirty_mnist/5%04d.png'%i
    im=Image.open(filepath)
    im=np.array(im)/255
    im=im.astype('float32')
    images_num.append(im)

np.save('C:/data/dacon/data2/test_set.npy', arr=images_num)

print('데이터 로드 종료', datetime.datetime.now()-time)


plt.imshow(images[0])
plt.imshow(images_num[0])
plt.imshow(Image.open('c:/data/dacon/data2/dirty_mnist/00000.png'))
plt.show()