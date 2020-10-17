from cv2 import cv2 as cv
import sys
import matplotlib.pyplot as plt
from main_ui import Ui_Opencvdl_HW1
from PyQt5.QtWidgets import QApplication, QDockWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

class App(Ui_Opencvdl_HW1):
    def __init__(self):
        super().__init__()

    def initUI(self):
        new_ui = Ui_Opencvdl_HW1()
        widget = QDockWidget()
        new_ui.setupUi(widget)
        button.clicked.connect(self.on_click)
        self.show()
    
    def on_click(self):    
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_Opencvdl_HW1()
    


    
    ex = App()
    sys.exit(app.exec_())