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

def plot_cm(model, x, y, target_names, filepath):
    cm = confusion_matrix(y, model.clf.predict(x))

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(filepath)

def plotImage(imgArr, label):
    imgArr = np.reshape(imgArr,(28, 28))
    plt.imshow(imgArr, cmap='gray')
    plt.title(label)
    plt.show()