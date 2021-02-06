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

train=list()
# train2=list()
# train3=list()
# train4=list()
# train5=list()
# for i in range(10000):
#     temp=Image.open('../data/dacon/data2/dirty_mnist_2nd/%05d.png'%i)
#     temp2=np.array(temp)
#     train.append(temp2)        
#     for j in range(10000, 20000):
#         temp=Image.open('../data/dacon/data2/dirty_mnist_2nd/%05d.png'%j)
#         temp2=np.array(temp)
#         train.append(temp2)
#         for k in range(20000, 30000):
#             temp=Image.open('../data/dacon/data2/dirty_mnist_2nd/%05d.png'%k)
#             temp2=np.array(temp)
#             train.append(temp2)
#             for l in range(30000, 40000):
#                 temp=Image.open('../data/dacon/data2/dirty_mnist_2nd/%05d.png'%l)
#                 temp2=np.array(temp)
#                 train.append(temp2)
#                 for m in range(40000, 50000):
#                     temp=Image.open('../data/dacon/data2/dirty_mnist_2nd/%05d.png'%m)
#                     temp2=np.array(temp)
#                     train.append(temp2)

# img_list=list()
# img=Image.open('../data/dacon/data2/dirty_mnist_2nd/00000.png')
# img_1=np.array(img)
# img_pd=pd.DataFrame(img_1)
# # img_list.append(img_1)
# # img_pd.to_csv('../data/dacon/data2/img.csv')

# img2=Image.open('../data/dacon/data2/dirty_mnist_2nd/00001.png')
# img2_1=np.array(img2)
# img_pd2=pd.DataFrame(img2_1)

# imageset=pd.concat([img_pd, img_pd2], axis=0)
# imageset.to_csv('../data/dacon/data2/img.csv')

# # train.to_csv('../data/dacon/data2/train_1.csv')
# # train2=train2.to_numpy()

# # print(train.shape)
# # print(train2.shape)

# imgs=pd.read_csv('../data/dacon/data2/img.csv')

# print(imgs.info())
# print(imgs.head())

# images=list()
# for i in range(50000):
#     filepath='../data/dacon/data2/dirty_mnist_2nd/%05d.png'%i
#     img=Image.open(filepath)
#     img=np.array(img)/255
#     img=img.astype('float32')
#     img=pd.DataFrame(img) 
#     images.append(img)

# imgsets=pd.concat(images)

# imgsets.to_csv('../data/dacon/data2/train.csv', index=False)

images_num=list()
for i in range(5000):
    filepath='../data/dacon/data2/test_dirty_mnist_2nd/5%04d.png'%i
    im=Image.open(filepath)
    im=np.array(im)/255
    im=im.astype('float32')
    images_num.append(im)

np.save(images_num, arr=images_num)