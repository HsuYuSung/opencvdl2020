

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cv2 import cv2 as cv
from scipy import signal
from scipy import misc

#3*3 Gassian filter
x, y = np.mgrid[-1:2, -1:2]

gaussian_kernel = np.exp(-(x**2+y**2))
#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
img = mpimg.imread('Chihiro.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
grad = signal.convolve2d(gray, gaussian_kernel, boundary='symm', mode='same') #卷積
plt.figure('Gaussian Blur')
plt.imshow(grad, cmap=plt.get_cmap('gray'))
plt.show()

