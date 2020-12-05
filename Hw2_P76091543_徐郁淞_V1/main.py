import os.path
import sys
from cv2 import cv2 as cv
import matplotlib.image as mpimg
from scipy import ndimage, misc, signal
from PIL import Image
from scipy.ndimage import filters

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from main_ui import Ui_DockWidget
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QLineEdit, QLabel


class window(QDockWidget, Ui_DockWidget):
    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        self.setupUi(self)
        self.on_binding_ui()

    def on_binding_ui(self):
        self.pushButton.clicked.connect(self.DrawContour1)
        self.pushButton_2.clicked.connect(self.DrawContour2)
        self.pushButton_3.clicked.connect(self.calibration)

    def DrawContour1(self):
        plt.close('all')

        im = cv.imread('Datasets/Q1_Image/coin01.jpg')
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(imgray, (15, 15), 0)
        ret, thresh = cv.threshold(gray_blur, 127, 255, 0)
        im2, contours1, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE)

        contours1.pop(0)
        cv.drawContours(im, contours1, -1, (0, 255, 0), 2)

        coin_count1 = len(contours1)
        text = 'There are ' + str(coin_count1) + ' coins in coin01.jpg'
        label = self.label_2
        label.setText(text)

        cv2plt = im[:,:,::-1]
        plt.imshow(cv2plt)
        plt.show()

    def DrawContour2(self):
        plt.close('all')

        im = cv.imread('Datasets/Q1_Image/coin02.jpg')
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(imgray, (15, 15), 0)
        ret, thresh = cv.threshold(gray_blur, 127, 255, 0)
        im2, contours2, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE)

        contours2.pop(0)
        cv.drawContours(im, contours2, -1, (0, 255, 0), 2)

        coin_count2 = len(contours2)
        text = 'There are ' + str(coin_count2) + ' coins in coin01.jpg'
        label = self.label_3
        label.setText(text)

        cv2plt = im[:,:,::-1]
        plt.imshow(cv2plt)
        plt.show()

    def calibration(self):
        plt.close('all')

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for i in range(1, 16):
            img_name = str(i) + '.bmp'
            fname = os.path.join('Datasets/Q2_Image', img_name)

            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (11, 8), corners2, ret)
                plt_img = img[:, :, ::-1]
                plt.figure(fname)
                plt.imshow(plt_img)

        plt.show()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = window()
    ui.show()

    sys.exit(app.exec_())
