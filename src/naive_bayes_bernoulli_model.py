from sklearn.naive_bayes import BernoulliNB
from base_model import BaseModel

class NaiveBayesBernoulliModel(BaseModel):
    """Perform BernoulliNB
    Example usage:                                                    
    > model = NaiveBayesBernoulliModel()                               
    > model.train(x_train, y_train)                                   
    > model.predict(x_eval)                                           
    """

    def __init__(self, step_size=0.2, max_iter=1e5, threshold=1e-5, verbose=False):
        BaseModel.__init__(self, step_size, max_iter, threshold, verbose)
        self.clf = BernoulliNB()

    def train(self, x, y):
        """fit classifier                                             
        Args:                                                         
            x: Training example inputs. Shape (m, n).                 
            y: Training example labels. Shape (m,).                   
        """
        # *** START CODE HERE ***                                     
        self.clf.fit(x, y)

        # *** END CODE HERE ***                                       

    def predict(self, x):
        """Make a prediction given new inputs x.                      
                m - number of samples in the set x 
                """
        # *** START CODE HERE ***                                                           
        return self.clf.predict(x)
        # *** END CODE HERE ***                                                             

