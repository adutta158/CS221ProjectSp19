import numpy as np
from sklearn.naive_bayes import BernoulliNB
import util.py

class NaiveBayesBernoulliModel(BaseModel):
    """Perform BernoulliNB
    Example usage:                                                    
    > model = NaiveBayesBernoulliModel()                               
    > model.train(x_train, y_train)                                   
    > model.predict(x_eval)                                           
    """

    def __init__(self, alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
one):
        BaseModel.__init__(self,alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        self.clf = BernoulliNB()
        clf.fit(x,y)
        BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=None)

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
                """
        # *** START CODE HERE ***                                                           
        return self.clf.predict(x)
        # *** END CODE HERE ***                                                             

