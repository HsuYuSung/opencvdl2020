
import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = '2020 Opencvdl2020 HW1'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        button = QPushButton('1.1 Load Image', self)
        button.setToolTip('1.1 Load Image')
        button.move(30,30)
        button.clicked.connect(self.on_click)
        button2 = QPushButton('1.2 Color Seperate', self)
        button2.setToolTip('')
        button2.move(30,80)
        button2.clicked.connect(self.color_seprate)
        button3 = QPushButton('1.3 Image Flipping', self)
        button3.setToolTip('')
        button3.move(30,130)
        button3.clicked.connect(self.image_flip)
        button4 = QPushButton('1.3 Blending', self)
        button4.setToolTip('')
        button4.move(30,180)
        button4.clicked.connect(self.blending)
        
        self.show()

    @pyqtSlot()
    def on_click(self):
        img = plt.imread("Uncle_Roger.jpg")
        plt.imshow(img)
        plt.show()
    
    def color_seprate(self):
        flower = cv.imread("Flower.jpg")
        # plt.imshow(flower)
        # plt.imshow(flower/255.0, cmap='Green')
        b,g,r = cv.split(flower)
        cv.imshow("b",r)
        cv.waitKey(0)

    def image_flip(self):
        img = plt.imread("Uncle_Roger.jpg")
        plt.figure(1)
        plt.imshow(img)
        plt.figure(2)
        img_flip = img[:, ::-1, :]
        plt.imshow(img_flip)
        plt.show()
    def blending(self):
        print(1)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())