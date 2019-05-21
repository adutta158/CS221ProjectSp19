import numpy as np
from sklearn.naive_bayes import GaussianNB
import util.py

class NaiveBayesGaussianModel(BaseModel):
    """Perform GaussianNB
    Example usage:
    > model = NaiveBayesGaussianModel()
    > model.train(x_train, y_train)
    > model.predict(x_eval)
    """

    def __init__(self, priors=None, var_smoothing=1e-09)
one):
        BaseModel.__init__(self,priors=None, var_smoothing=1e-09)
        self.clf_pf = GaussianNB()

    def train(self, x, y):
        """fit classifier
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.clf_pf.partial_fit(x,y,np.unique(y))
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
        return self.clf_pf.predict(x)
        # *** END CODE HERE ***
