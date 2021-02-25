import cv2
import numpy as np
import matplotlib.pyplot as plt

# 데이터 크기 256, 256, 1
# for i in range(5000):
#     img=cv2.imread('c:/data/dacon/data2/dirty_mnist/%05d.png'%i)
#     img=np.array(img)
#     img=np.where((img<255) & (img!=0), 0, 1)
#     cv2.imwrite('c:/data/dacon/data2/dirty_mnist_train/%05d.png'%i, img)