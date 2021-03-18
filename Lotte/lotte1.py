from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2

from PIL import Image


imgset = list()
for i in range(72000):
    img = cv2.imread(
        'c:/LPD_competition/test/%s.jpg'%i
    )
    img = cv2.resize(img, (128, 128))
    img = np.array(img)/255.
    imgset.append(img)

imgset = np.array(imgset)

np.save(
    'c:/LPD_competition/test.npy', arr = imgset
)