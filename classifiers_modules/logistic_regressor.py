import numpy as np

class LogisticRegressor():
    def __init__(self, weights, bias, epochs, ) -> None:
        self.weights = weights
        self.bias    = bias
        self.epochs = epochs

        self.dot_mult = np.dot(weights, bias)
        self.sigmoid  = np.divide(1, 1 + np.exp(-self.dot_mult))
        self.epochs = epochs

    def loss(self,):
        """
            purpose: know how good our model is working?
            for logistic regressor, it is cross-entropy loss.
        """

    def optimizer(self,):
        """
            
        """

    def fit(self, X, y):
        """
            purpose: update weight and bias with an optimizer fuction like
            SGD
            hint: iterate over each epoch
        """
        print("Hola")

    def predict(self, X):
        """
        purpose: use the learned weights to make predictions on new data.
        
        """