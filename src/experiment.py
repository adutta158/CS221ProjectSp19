import os
import numpy as np
import util
from logistic_regression_model import LogisticRegressionModel
from majority_classifier_model import MajorityClassifierModel
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools

seed = 42 # for reproducibility

def experiment(model, x_train, y_train, x_dev, y_dev, model_name, class_names):
    # Train classifier
    model.train(x_train, y_train)
    c = len(class_names)
    y_pred = model.predict(x_dev)
    loss = log_loss(util.one_hot(y_dev, c), util.one_hot(y_pred, c))

    print('Accuracy = ' + str(accuracy_score(util.one_hot(y_dev, c), util.one_hot(y_pred, c))))
    print('f1 score = ' + str(f1_score(util.one_hot(y_dev, c), util.one_hot(y_pred, c), average='micro')))
    print('Log Loss = ' + str(loss))

    print(classification_report(util.one_hot(y_dev, c), util.one_hot(y_pred, c), target_names=class_names))

    plt.rcParams.update({'font.size': 6})
    cm = confusion_matrix(y_dev, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f').replace('.00', '').replace('0.', '.'),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig("output/cm_" + model_name)

    return loss
    # *** END CODE HERE ***


# NOTE: For neural network we will to create a new experiment.py
if __name__ == '__main__':
    print('Starting...')
    print('Load data')
    x, y, classes = util.load_dataset()
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)
    x_train, x_dev, y_train, y_dev = train_test_split(x_temp, y_temp, test_size=0.25, random_state = seed)

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
            for i in range(0,100):
                util.plotImage(x_dev[i, :], str(i))
                print(classes[int(y_dev[i])])

        elif i == 3:
            # Logistic Regression
            print('Running multiclass logistic regression model')
            lr_model = LogisticRegressionModel(verbose = True)
            loss = experiment(lr_model, x_train, y_train, x_dev, y_dev, 'logistic_regression', classes)

