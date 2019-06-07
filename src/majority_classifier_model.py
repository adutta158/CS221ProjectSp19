from base_model import BaseModel
import scipy
import util
import numpy as np


class MajorityClassifierModel(BaseModel):
    """Perform majority class prediction

    Example usage:
        > model = MajorityClassifierModel()
        > model.train(x_train, y_train)
        > model.predict(x_eval)
    """

    def __init__(self, step_size=0.2, max_iter=1e5, threshold=1e-5, verbose=False):
        BaseModel.__init__(self, step_size, max_iter, threshold, verbose)
        self.majorityClass = 0
        self.C = None

    def train(self, x, y):
        """fit classifier

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.C = int(y.max() + 1)


        self.majorityClass = scipy.stats.mode(y, axis=None)[0]

        if self.verbose:
            print("Inside MajorityClassifierModel.train, majority class = {0}".format(self.majorityClass))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

                m - number of samples in the set x
                n - number of features
                C - number of classes

                Args:
                    x: Inputs of shape (m, n).

                Returns:
                    Outputs of shape (m, 1).
        """
        # *** START CODE HERE ***
        y = np.array([self.majorityClass for i in range(x.shape[0])])
        return y
        # *** END CODE HERE ***
