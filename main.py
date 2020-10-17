
import sys
import matplotlib.pyplot as plt
from main_ui import Ui_Opencvdl_HW1
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QDockWidget



def load_image():
    img = plt.imread("Uncle_Roger.jpg")
    plt.imshow(img)
    plt.show()

def Color_Separation():
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QDockWidget()
    #test
    ui = Ui_Opencvdl_HW1()
    ui.setupUi(widget)
    
    ui.pushButton.clicked.connect(load_image)
    ui.pushButton_2.clicked.connect(Color_Separation)

    widget.show()
    ex = ui

    sys.exit(app.exec_())