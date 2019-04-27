import os
import numpy as np
import util
from logistic_regression_model import LogisticRegressionModel
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

seed = 42 # for reproducibility

def experiment(model, x_train, y_train, x_dev, y_dev, model_name, class_names):
    # Train classifier
    model.train(x_train, y_train)
    y_pred = model.predict(x_dev)
    loss = log_loss(util.one_hot(y_dev), y_pred)
    y_hot = util.one_hot(y_dev)
    print('Accuracy = ' + str(accuracy_score(util.one_hot(y_dev), y_pred)))
    print('f1 score = ' + str(f1_score(util.one_hot(y_dev), y_pred, average='micro')))
    print('Log Loss = ' + str(loss))

    print(classification_report(util.one_hot(y_dev), y_pred, target_names=class_names))

    return loss
    # *** END CODE HERE ***


# NOTE: For neural network we will to create a new experiment.py
if __name__ == '__main__':
    print('Starting...')
    print('Load data')
    x, y, classes = util.load_dataset()
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size = 0.2, random_state = seed)
    x_train, x_dev, y_train, y_dev = train_test_split(x_temp, y_temp, test_size=0.25, random_state = seed)

    # BaseLine
    print('Running baseline model')
    baseline_model = LogisticRegressionModel(verbose = True)
    loss = experiment(baseline_model, x_train, y_train, x_dev, y_dev, 'baseline', classes)
    util.plot_cm(baseline_model, x_dev, y_dev, classes, './output/CM_Baseline.PNG')

