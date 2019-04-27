from sklearn.linear_model import LogisticRegression
from base_model import BaseModel
import util


class LogisticRegressionModel(BaseModel):
    """Perform logistic regression

    Example usage:
        > model = LogisticRegressionModel()
        > model.train(x_train, y_train)
        > model.predict(x_eval)
    """

    def __init__(self, step_size=0.2, max_iter=1e5, threshold=1e-5, verbose=False, alpha=0, seed = None, penalty = 'l2', class_weight = None):
        BaseModel.__init__(self, step_size, max_iter, threshold, verbose)
        self.clf = LogisticRegression(solver='lbfgs', multi_class='auto')
        self.C = None

    def train(self, x, y):
        """fit classifier

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.C = int(y.max() + 1)
        self.clf.fit(x, y.ravel())

        if self.verbose:
            print("Inside LogisticRegressionModel.train, number of classes = {0}".format(self.C))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

                m - number of samples in the set x
                n - number of features
                C - number of classes

                Args:
                    x: Inputs of shape (m, n).

                Returns:
                    Outputs of shape (m,C).
        """
        # *** START CODE HERE ***
        return util.one_hot(self.clf.predict(x), self.C)
        # *** END CODE HERE ***
