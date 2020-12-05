import sys
from main_ui import Ui_DockWidget
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import regularizers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QDockWidget


class window(QDockWidget, Ui_DockWidget):
    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        (self.x_train, y_train), (self.x_test, y_test) = datasets.cifar10.load_data()
        # 将类向量转换为二进制类矩阵。
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)
        self.cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.reload_model = keras.models.load_model(
            'saved_models/keras_cifar10_trained_model.h5')

        print('please wait the model reload')
        self.reload_model.fit(self.x_test, self.y_test)

        self.probability_model = tf.keras.Sequential(
            [self.reload_model, tf.keras.layers.Softmax()])
        self.predictions = self.probability_model.predict(self.x_test)

        self.setupUi(self)
        self.data()
        self.on_binding_ui()
        
    def data(self):
        pass
        

    def on_binding_ui(self):
        self.pushButton.clicked.connect(self.ShowTrainImage)
        self.pushButton_2.clicked.connect(self.hyper)
        self.pushButton_3.clicked.connect(self.model_struct)
        self.pushButton_4.clicked.connect(self.acc_image)
        self.pushButton_5.clicked.connect(self.predict)


    def ShowTrainImage(self):
        plt.close()
        r0 = [np.random.randint(0, 50000) for _ in range(10)]

        a = list()
        index = list()
        index_class = list()
    

        for _ in range(10):
            a.append(r0[_])
            index.append(np.argmax(self.y_train[a[_]]))
            index_class.append(self.cifar_classes[index[_]])

        print(index)
        print(index_class)

        f, axarr = plt.subplots(2,5)

        for i in range(5):
            img = self.x_train[r0[i]]
            axarr[0, i].imshow(img)

        for i in range(5,10):
            img = self.x_train[r0[i]]
            axarr[1, i-5].imshow(img)
        
        plt.show()

    
    def hyper(self):
        batch_size = 32
        lr = 0.0001
        print('batch_size = ', batch_size)
        print('lr = ', lr)
        print('opt = RMSprop')
        

    def model_struct(self):
        model = Sequential()
        weight_decay = 0.0005
        x_shape = [32,32,3]
        num_classes = 10

        model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.summary()


    def acc_image(self):
        plt.close()
        img = plt.imread('accplot.png')
        plt.imshow(img)
        plt.show()
    
    def predict(self):
        
        predictions = self.predictions

        def plot_image(i, predictions_array, true_label, img):
            true_label, img = true_label[i], img[i]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            plt.imshow(img, cmap=plt.cm.binary)

            predicted_label = np.argmax(predictions_array)
            true_label = np.argmax(true_label)
            if predicted_label == true_label:
                color = 'blue'
            else:
                color = 'red'

            plt.xlabel("{} {:2.0f}% ({})".format(self.cifar_classes[predicted_label],
                                            100*np.max(predictions_array),
                                            self.cifar_classes[true_label]),
                                            color=color)

        def plot_value_array(i, predictions_array, true_label):
            true_label = true_label[i]
            true_label = np.argmax(true_label)
            plt.grid(False)
            plt.xticks(range(10))
            plt.yticks([])
            thisplot = plt.bar(range(10), predictions_array, color="#777777")
            plt.ylim([0, 1])
            predicted_label = np.argmax(predictions_array)

            thisplot[predicted_label].set_color('red')
            thisplot[true_label].set_color('blue')


        i = self.lineEdit.text()
        if i:
            i = int(i)
        else:
            i = 0

        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)


        plot_image(i, predictions[i], self.y_test, self.x_test)
        plt.subplot(1,2,2)
        plot_value_array(i, predictions[i],  self.y_test)
        plt.show()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = window()
    ui.show()

    sys.exit(app.exec_())

