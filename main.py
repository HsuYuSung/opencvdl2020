
import sys
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from main_ui import Ui_Opencvdl_HW1
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget




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

    def Median_filter(self):
        img = cv.imread('Cat.png')
        median = cv.medianBlur(img, 7)
        plt_img = img [:,:,::-1]
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



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = window()
    ui.show()

    sys.exit(app.exec_())