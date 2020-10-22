
import sys
from cv2 import cv2 as cv
import matplotlib.image as mpimg
from scipy import ndimage, misc, signal
from PIL import Image
from scipy.ndimage import filters

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from main_ui import Ui_Opencvdl_HW1
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QLineEdit

class window(QDockWidget ,Ui_Opencvdl_HW1):
    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        self.setupUi(self)
        self.on_binding_ui()
        
    def on_binding_ui(self):
        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.Color_Separation)
        self.pushButton_3.clicked.connect(self.Image_Flipping)
        self.pushButton_4.clicked.connect(self.Image_Tracebar)
        self.pushButton_5.clicked.connect(self.Median_filter)
        self.pushButton_6.clicked.connect(self.Gaussian_filter)
        self.pushButton_7.clicked.connect(self.Bilateral_filter)
        self.pushButton_8.clicked.connect(self.Gaussian_blur)
        self.pushButton_9.clicked.connect(self.sobel_x)
        self.pushButton_10.clicked.connect(self.sobel_y)
        self.pushButton_11.clicked.connect(self.magnitude)
        self.pushButton_12.clicked.connect(self.text)

    def text(self):
        rot_n = 0
        scale_n = 1
        x_n = 0
        y_n = 0
        x_old = 160
        y_old = 84

        rot = self.Tx.text()
        scale = self.Tx_2.text()
        transform_x = self.Tx_3.text()
        transform_y = self.Tx_4.text()
        if rot:
            rot_n = float(rot)
        if scale:
            scale_n = float(scale)
        if transform_x:
            x_n = float(transform_x)
        if transform_y:
            y_n = float(transform_y)

        img = cv.imread('Parrot.png')

        rows, cols, _ = img.shape

        #translate positon
        M = np.float32([[1,0,x_n],[0,1,y_n]])
        tra_dst = cv.warpAffine(img,M,(cols,rows))

        #rotate and scale
        M = cv.getRotationMatrix2D((x_old + x_n, y_old + y_n),rot_n, scale_n)
        dst = cv.warpAffine(tra_dst,M,(cols, rows))

        dst_plt = dst[:,:,::-1]
        plt.imshow(dst_plt)
        plt.show()
        
    def load_image(self):
        img = plt.imread('Uncle_Roger.jpg')
        plt.imshow(img)
        plt.show()

    def Color_Separation(self):
        img = plt.imread('Flower.jpg').copy()
        r = img.copy()
        g = img.copy()
        b = img.copy()

        r[:,:,1] = 0
        r[:,:,2] = 0
        g[:,:,0] = 0
        g[:,:,2] = 0
        b[:,:,0] = 0
        b[:,:,1] = 0
        plt.figure("origin")
        plt.imshow(img)
        plt.figure("r")
        plt.imshow(r)
        plt.figure("g")
        plt.imshow(g)
        plt.figure("b")
        plt.imshow(b)
        plt.show()
    
    def Image_Flipping(self):
        img = plt.imread('Uncle_Roger.jpg')
        plt.figure("origin")
        plt.imshow(img)
        img_reverse = img[:,::-1,:]
        plt.figure("reverse")
        plt.imshow(img_reverse)
        plt.show()
    
    def Image_Tracebar(self):
        img = plt.imread('Uncle_Roger.jpg')
        img_reverse = img[:,::-1,:]

        # plt.imshow(img, alpha=0.4)
        axbar = plt.axes([0.1, 0.9, 0.8, 0.03])
        sbar = Slider(axbar, 'Blend', 0, 1, valinit=0.5)
        
        fig = plt.subplot()
        plt.imshow(img_reverse, alpha = 0.5)
        plt.imshow(img, alpha = 0.5)

        def image(val):
            plt.cla()
            plt.imshow(img_reverse, alpha = 1 - val)
            plt.imshow(img, alpha = val)
            plt.show()
            
        def update(val):
            value = sbar.val
            image(value)

        sbar.on_changed(update)
        plt.show()

    def Median_filter(self):
        img = cv.imread('Cat.png')
        median = cv.medianBlur(img, 7)
        plt_img = img[:,:,::-1]
        plt.figure('origin')
        plt.imshow(plt_img)
        plt_median = median[:,:,::-1]
        plt.figure('median')
        plt.imshow(plt_median)
        plt.show()

    def Gaussian_filter(self):
        img = cv.imread('Cat.png')
        gaussian_img = cv.GaussianBlur(img,(3,3),0)
        plt_img = img [:,:,::-1]
        plt.figure('origin')
        plt.imshow(plt_img)
        plt_gaussian_img = gaussian_img[:,:,::-1]
        plt.figure('Gaussian')
        plt.imshow(plt_gaussian_img)
        plt.show()

    def Bilateral_filter(self):
        img = cv.imread('Cat.png')
        blur = cv.bilateralFilter(img,9,90,90)

        plt_img = img [:,:,::-1]
        plt.figure('origin')
        plt.imshow(plt_img)
        plt_blur = blur[:,:,::-1]
        plt.figure('Bilateral')
        plt.imshow(plt_blur)
        plt.show()
    
    def Gaussian_blur(self):
        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]

        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        img = plt.imread('Chihiro.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grad = signal.convolve2d(gray, gaussian_kernel, boundary='symm', mode='same') #卷積
        plt.figure('Gaussian Blur')
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.show()

    def sobel_x(self):
        pil_img = Image.open('Chihiro.jpg').convert('L')
        img = np.array(pil_img)

        result = np.zeros(img.shape)
        filters.sobel(img, 1, result)
        plt.imshow(result, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def sobel_y(self):
        pil_img = Image.open('Chihiro.jpg').convert('L')
        img = np.array(pil_img)

        result = np.zeros(img.shape)
        filters.sobel(img, 0, result)
        plt.imshow(result, cmap='gray', vmin=0, vmax=255)
        plt.show()
    
    def magnitude(self):
        pil_img = Image.open('Chihiro.jpg').convert('L')
        img = np.array(pil_img)
        
        x_mask = np.zeros(img.shape)
        y_mask = np.zeros(img.shape)
        filters.sobel(img, 1, x_mask)
        filters.sobel(img, 1, y_mask)
        result = np.hypot(x_mask, y_mask)
        plt.imshow(result, cmap='gray', vmin=0, vmax=255)
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = window()
    ui.show()

    sys.exit(app.exec_())