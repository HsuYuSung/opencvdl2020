
import sys
import matplotlib.pyplot as plt
from main_ui import Ui_Opencvdl_HW1
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow




class mainwindow(QMainWindow ,Ui_Opencvdl_HW1):
    def __init__(self):
        super(mainwindow, self).__init__()

    def setupUi(self, Ui_Opencvdl_HW1):
        self.on_binding_ui()
        
    
    def on_binding_ui(self):
        self.pushButton.clicked.connect(self.load_image)

    def load_image(self):
        img = plt.imread('Uncle_Roger.jpg')
        plt.imshow(img)
        plt.show()
        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    application = mainwindow()
    widget = QDockWidget()
    application.setupUi(widget)
    # ui = Ui_Opencvdl_HW1()
    # widget = QDockWidget()
    # ui.setupUi(widget)
    # widget.show()
    # u
    ex = application
    sys.exit(app.exec_())