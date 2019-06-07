import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools


def load_dataset(ipc = 20000):
    """Load dataset from .npy files
    Args:
        ipc: number of images per class to take to build dataset
    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
        classes: list of doodle classes
    """
    files = os.listdir("..\\data")
    ind = 0
    xs = []
    ys = []
    classNames = []
    for file in files:
        fileSplit = file.split('.')
        print('--Loading ' + fileSplit[0][18:] + ' data.')
        classNames.append(fileSplit[0][18:])
        x = np.load("..\\data\\" + file)
        x = x.astype('float32')/255
        xs.append(x[0:ipc, :])
        y = np.array([float(ind) for i in range(ipc)])
        ys.append(y.reshape(ipc, 1))
        ind += 1

    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs.reshape(xs.shape[0]*xs.shape[1], xs.shape[2])
    ys = ys.reshape(ys.shape[0]*ys.shape[1], ys.shape[2])
    return xs, ys, classNames

def one_hot(y, C=None):
    m = y.shape[0]
    if C is None:
        C = int(y.max() + 1)

    return np.squeeze(np.eye(C)[y.astype('int')])

def plotImage(imgArr, label):
    imgArr = np.reshape(imgArr,(28, 28))
    plt.imshow(imgArr, cmap='gray')
    plt.title(label)
    plt.show()
