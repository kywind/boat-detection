import cv2
import numpy as np

img = cv2.imread('./map_2000/2010.png')
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if np.all(img[i][j] == 0):
            img[i][j] = np.array([255, 255, 255])

cv2.imwrite('./map_2000/2010_white.png', img)