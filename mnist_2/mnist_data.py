import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

number = 5000

for a in np.arange(50000, 55000):
    Img = cv2.imread('c:/data/dacon/data2/test_dirty_mnist/' + str(a).zfill(5) + '.png', cv2.IMREAD_GRAYSCALE)
    #254보다 작고 0이아니면 0으로 만들어주기
    img2 = np.where((Img <= 254) & (Img != 0), 0, Img)
    # 이미지 팽창
    img2 = cv2.dilate(img2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    # 블러 적용, 노이즈 제거
    img2 = cv2.medianBlur(src=img2, ksize= 5)
    # canny
    # img2 = cv2.Canny(img2, 30, 70)

    cv2.imwrite('c:/data/dacon/data2/dirty_mnist_test/'+ str(a).zfill(5) + '.png', img2)

print('done')