
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

    def load_image(self):
        img = plt.imread('Uncle_Roger.jpg')
        plt.imshow(img)
        plt.show()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = window()
    ui.show()
    
    sys.exit(app.exec_())