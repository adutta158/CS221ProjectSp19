import numpy as np
from sklearn.naive_bayes import MultinomialNB
import util.py

class NaiveBayesMultinomialModel(BaseModel):
    """Perform logistic regression
    Example usage:
    > model = NaiveBayesMultinomialModel()
    > model.train(x_train, y_train)
    > model.predict(x_eval)
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None)
one):
        BaseModel.__init__(self, alpha=1.0, fit_prior=True, class_prior=None)
        self.clf = GaussianNB()
        clf.fit(x,y)

    def train(self, x, y):
        """fit classifier
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.clf.fit(self, x, y, None)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
                m - number of samples in the set x
                n - number of features
                C - number of classes
                Args:
                    x: Inputs of shape (m, n).
                    Returns:
                    Outputs of shape (m, 1)
        """
        # *** START CODE HERE ***
        return self.clf.predict(x)
        # *** END CODE HERE ***

