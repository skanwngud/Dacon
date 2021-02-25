# 라이브러리 임포트
import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('c:/data/dacon/data2/dirty_mnist_train/00001.png')

b,g,r=cv2.split(image) # bgr 컬러포맷을
image=cv2.merge([r, g, b]) # rgb 로 바꿔줌

blur=cv2.GaussianBlur(
    image,
    ksize=(9, 9),
    sigmaX=0
)

ret, thresh1=cv2.threshold(
    blur,
    127, 255,
    cv2.THRESH_BINARY
)

edged=cv2.Canny(
    blur,
    10, 250
)

kernel=cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (7,7)
)

closed=cv2.morphologyEx(
    edged,
    cv2.MORPH_CLOSE,
    kernel
)

	
contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0


# contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
# cv2.imshow('contours_image', contours_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours_xy = np.array(contours)
contours_xy.shape

x_min, x_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)

print(x_min)
print(x_max)
 
# y의 min과 max 찾기
y_min, y_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)

print(y_min)
print(y_max)

x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min

img_trim = image[y:y+h, x:x+w]
cv2.imwrite('c:/data/dacon/data2/dirty_mnist_train/org_trim.jpg', img_trim)
org_image = cv2.imread('org_trim.jpg')

cv2.imshow('org_image', org_image)
cv2.waitKey(0)
cv2.destroyAllWindows()