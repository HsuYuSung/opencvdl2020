import numpy as np


y_train = np.zeros((50000,1))

r0 = [np.random.randint(0, 50000) for _ in range(10)]
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(r0)
a = list()
index = list()
index_class = list()
a.append(r0[0])
a.append(r0[1])
print('train', int(y_train[a[1]]))

    # index.append(y_train[a[_]].astype(int))
    # print(index)
    # index_class.append(cifar_classes[index[_]])

