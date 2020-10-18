
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from main_ui import Ui_Opencvdl_HW1
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons




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

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = window()
    ui.show()

    sys.exit(app.exec_())