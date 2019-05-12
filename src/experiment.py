import os
import numpy as np
import util
from logistic_regression_model import LogisticRegressionModel
from majority_classifier_model import MajorityClassifierModel
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import pickle

seed = 42 # for reproducibility

def experiment(model, x_train, y_train, x_dev, y_dev, model_name, class_names):
    # Train classifier
    model.train(x_train, y_train)
    c = len(class_names)
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_dev)
    loss = log_loss(util.one_hot(y_dev, c), util.one_hot(y_pred, c))
    loss_train = log_loss(util.one_hot(y_train, c), util.one_hot(y_train_pred, c))

    print('Train Accuracy = ' + str(accuracy_score(util.one_hot(y_train, c), util.one_hot(y_train_pred, c))))
    print('Train Log Loss = ' + str(loss_train))
    print('Test Accuracy = ' + str(accuracy_score(util.one_hot(y_dev, c), util.one_hot(y_pred, c))))
    print('Test Log Loss = ' + str(loss))
    print(classification_report(util.one_hot(y_dev, c), util.one_hot(y_pred, c), target_names=class_names))

    plt.rcParams.update({'font.size': 8})
    cm = confusion_matrix(y_dev, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title("Baseline matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, range(0, 22))#, rotation=90)
    plt.yticks(tick_marks, range(0, 22))

    thresh = cm.max() / 2.
    '''for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f').replace('.00', '').replace('0.', '.'),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")'''

    plt.ylabel('True label index')
    plt.xlabel('Predicted label index')
    plt.tight_layout()
    fig.savefig("output/cm_" + model_name)

    return loss
    # *** END CODE HERE ***

def experiment2(y_pred, y_dev, model_name):
    # Train classifier
    c = 22
    loss = log_loss(util.one_hot(y_dev, c), util.one_hot(y_pred, c))

    print('Accuracy = ' + str(accuracy_score(util.one_hot(y_dev, c), util.one_hot(y_pred, c))))
    print('f1 score = ' + str(f1_score(util.one_hot(y_dev, c), util.one_hot(y_pred, c), average='micro')))
    print('Log Loss = ' + str(loss))

    print(classification_report(util.one_hot(y_dev, c), util.one_hot(y_pred, c)))

    plt.rcParams.update({'font.size': 8})
    cm = confusion_matrix(y_dev, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title("Baseline matrix")
    plt.colorbar()
    tick_marks = np.arange(c)
    plt.xticks(tick_marks, range(0, 22))#, rotation=90)
    plt.yticks(tick_marks, range(0, 22))

    thresh = cm.max() / 2.
    '''for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f').replace('.00', '').replace('0.', '.'),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")'''

    plt.ylabel('True label index')
    plt.xlabel('Predicted label index')
    plt.tight_layout()
    fig.savefig("output/cm_" + model_name)

    return loss
    # *** END CODE HERE ***

# NOTE: For neural network we will to create a new experiment.py
if __name__ == '__main__':
    print('Starting...')
    print('Load data')

    # Uncomment code below to load new data.
    '''x, y, classes = util.load_dataset()
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)
    x_train, x_dev, y_train, y_dev = train_test_split(x_temp, y_temp, test_size=0.25, random_state = seed)

    pickle.dump(x_train, open("x_train.p", "wb"))
    pickle.dump(y_train, open("y_train.p", "wb"))
    pickle.dump(x_dev, open("x_dev.p", "wb"))
    pickle.dump(y_dev, open("y_dev.p", "wb"))
    pickle.dump(x_test, open("x_test.p", "wb"))
    pickle.dump(y_test, open("y_test.p", "wb"))
    pickle.dump(classes, open("classes.p", "wb"))'''

    x_train = pickle.load(open("x_train.p", "rb"))
    y_train = pickle.load(open("y_train.p", "rb"))
    x_dev = pickle.load(open("x_dev.p", "rb"))
    y_dev = pickle.load(open("y_dev.p", "rb"))
    x_test = pickle.load(open("x_test.p", "rb"))
    y_test = pickle.load(open("y_test.p", "rb"))
    classes = pickle.load(open("classes.p", "rb"))

    while(1):
        print("\n\n---------------------------------------")
        print("1. Run baseline model")
        print("2. Run oracle model")
        print("3. Run logistic regression model")
        print("0. Exit")
        print("---------------------------------------")
        i = int(input("Enter choice: "))

        if i == 1:
            # BaseLine
            print('Running baseline model')
            baseline_model = MajorityClassifierModel(verbose = True)
            loss = experiment(baseline_model, x_train, y_train, x_dev, y_dev, 'baseline', classes)

        if i == 2:
            # Oracle
            print('Running oracle model')
            y_true = []
            for i in range(22):
                for j in range(30):
                    y_true.append(i)
            y_pred = y_true[:]
            y_pred[30*15] = 3
            y_pred[(30 * 15) + 1] = 7
            y_pred[(30 * 15) + 2] = 7
            y_pred[(30 * 14) + 0] = 19
            y_pred[(30 * 17) + 0] = 1
            y_pred[(30 * 17) + 1] = 1
            y_pred[(30 * 17) + 2] = 1
            y_pred[(30 * 17) + 3] = 1
            y_pred[(30 * 17) + 4] = 1
            y_pred[(30 * 17) + 5] = 1
            y_pred[(30 * 17) + 6] = 10
            y_pred[(30 * 3) + 0] = 15
            y_pred[(30 * 10) + 0] = 8
            y_pred[(30 * 10) + 1] = 9
            y_pred[(30 * 10) + 2] = 0
            y_pred[(30 * 10) + 3] = 0
            y_pred[(30 * 16) + 0] = 1
            y_pred[(30 * 16) + 1] = 5
            y_pred[(30 * 3) + 0] = 8
            y_pred[(30 * 12) + 0] = 3
            y_pred[(30 * 9) + 0] = 3
            y_pred[(30 * 20) + 0] = 17
            experiment2(np.array(y_pred), np.array(y_true), 'oracle')
            print(y_pred)
            print(y_true)

        elif i == 3:
            # Logistic Regression
            print('Running multiclass logistic regression model')
            for c in [0.01, 0.1, 0.5, 1]:
                print('Running with hyperparameter C = ' + str(c))
                lr_model = LogisticRegressionModel(verbose = True, C = c)
                loss = experiment(lr_model, x_train, y_train, x_dev, y_dev, 'logistic_regression_C_' + str(c).replace('.', '_'), classes)

