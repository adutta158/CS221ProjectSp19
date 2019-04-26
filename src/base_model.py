class BaseModel(object):
    """Base class for (classification) models."""

    def __init__(self, step_size=0.2, max_iter=100, threshold=1e-5, verbose=False):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.threshold = threshold
        self.verbose = verbose

    def train(self, x, y):
        """Train the classification using x and y

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        raise NotImplementedError('Subclass of BaseModel must implement train method.')

    def predict(self, x):
        """Make a prediction given new inputs x. Specifically, provide predicted probabilities for each element of x
        belonging to a class c

        m - number of samples in the set x
        n - number of features
        C - number of classes

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,C).
        """
        raise NotImplementedError('Subclass of Base must implement predict method.')
