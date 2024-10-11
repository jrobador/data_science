import numpy as np

class LogisticRegressor():
    def __init__(self, weights, bias, epochs, lr) -> None:
        self.weights = weights
        self.bias    = bias
        self.epochs = epochs
        self.lr = lr

        self.dot_mult = np.dot(weights, bias)
        self.sigmoid  = np.divide(1, 1 + np.exp(-self.dot_mult))


    def loss(self,):
        """
            purpose: know how good our model is working?
            for logistic regressor, it is cross-entropy loss.
        """

    def sgd_optimizer(self,):
        """
           purpose: adjust the weights and bias after each iteration
             using the gradient descent formula. 
        """

    def fit(self, X, y):
        """
            purpose: update weight and bias with an optimizer fuction like
            SGD
        """

    def predict(self, X):
        """
        purpose: use the learned weights to make predictions on new data.
        """