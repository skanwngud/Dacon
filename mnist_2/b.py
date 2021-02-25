# 글자 인식 시도
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

img=Image.open(r'c:/data/dacon/data2/dirty_mnist_train/00000.png')

'''
width, height=img.size
print(img.size) # (256, 256)

x1, y1, x2, y2=img.getbbox()
print(x1, y1, x2, y2)

img1=img.crop((22, 13, 244, 233))

plt.imshow(img1)
# plt.imshow(img)
plt.show()
'''

for i in range(50000):
    large=cv2.imread('c:/data/dacon/data2/dirty_mnist_train/%05d.png'%i)
    rgb=cv2.pyrDown(large)
    small=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    grad=cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw=cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected=cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # RETR_CCOMP 대신 RETR_EXTERNAL 사용? 뭔 소리야 이게
    contours, hieracrchy=cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask=np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x,y,w,h=cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w]=0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r=float(cv2.countNonZero(mask[y:y+h, x:x+w]))/(w*h)
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x,y), (x+w-1, y+h-1), (0, 255, 0), 2)

    cv2.imshow('rects', rgb)
    cv2.waitKey()
