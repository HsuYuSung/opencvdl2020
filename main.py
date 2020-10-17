
import sys
import matplotlib.pyplot as plt
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
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = window()
    ui.show()

    sys.exit(app.exec_())