from cv2 import cv2 as cv
import sys
import matplotlib.pyplot as plt
from main_ui import Ui_Opencvdl_HW1
from PyQt5.QtWidgets import QApplication, QDockWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot


# class App(Ui_Opencvdl_HW1):
#     def setupUi(self, Opencvdl_HW1):
#         super().setupUi(Opencvdl_HW1)
#     def retranslateUi(self, Opencvdl_HW1):
#         super().retranslateUi(Opencvdl_HW1)
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_Opencvdl_HW1()
    widget = QDockWidget()
    ui.setupUi(widget)
    
    sys.exit(app.exec_())