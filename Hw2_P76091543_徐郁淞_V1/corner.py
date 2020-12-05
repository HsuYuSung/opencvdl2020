import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt

filename = 'Datasets/Q2_Image/1.bmp'
img = cv.imread(filename)
print(img.shape)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.001*dst.max()] = [0, 0, 255]
plt_img = img[:,:,::-1]

plt.imshow(plt_img)
plt.show()


