
import sys
import matplotlib.pyplot as plt
from main_ui import Ui_Opencvdl_HW1
from PyQt5.QtWidgets import QApplication, QDockWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot


if __name__ == '__main__':
    app = QApplication(sys.argv)
    new_ui = Ui_Opencvdl_HW1()
    widget = QDockWidget()
    new_ui.setupUi(widget)

    ui = Ui_Opencvdl_HW1()
    widget.show()
    ex = new_ui
    sys.exit(app.exec_())