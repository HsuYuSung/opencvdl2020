from cv2 import cv2 as cv
import sys
import matplotlib.pyplot as plt


img = plt.imread("Flower.jpg").copy()
b = img.copy()
g = img.copy()
r = img.copy()

b[:,:,0] = 0.0
b[:,:,1] = 0.0
plt.figure('b')
plt.imshow(b)

g[:,:,0] = 0.0
g[:,:,2] = 0.0
plt.figure('g')
plt.imshow(g)


r[:,:,1] = 0.0
r[:,:,2] = 0.0
plt.figure('r')
plt.imshow(r)




plt.show()