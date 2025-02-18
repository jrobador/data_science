#%%

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# %%
def make_custom_regression(n_samples: int, n_features: int, noise: int=0, random_state: int = 37):
    # La regresion es de tipo y = \sum (mx + b)

    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.rand(n_samples, n_features)

    coefs = np.random.rand(n_features)

    Y = X @ coefs

    if noise > 0:
        Y += np.random.rand(n_features) + noise

    # Y = X @ coefs + noise
    # For samples = 10, features = 1, noise = 1:
    # Y = [10, 1] [1] + [10]
    return X, Y, coefs

n_features = 10
X, Y, coefs = make_custom_regression(n_samples=3000, n_features=n_features, noise=1)

X_np = np.array(X)
Y_np = np.array(Y)
coefs = np.array(coefs)
print (X_np.shape, Y_np.shape, coefs.shape)
if n_features == 2:
    plt.scatter(X,Y)


def linear(inp, weights, bias):
    #Input: (batch_size, input_dim)
    #Weights: (input_dim, output_dim)
    #Bias: (output_dim)

    if inp.shape[1] != weights.shape[0]:
        raise RuntimeError("Las dimensiones enter Input y Weights es incongruente")
    if bias.shape[0] != weights.shape[1]:
        raise RuntimeError("El bias no tiene la misma dimension que la salida")
    
    # Multiplicación matricial: (batch_size, input_dim) @ (input_dim, output_dim) -> (batch_size, output_dim)
    y = inp @ weights
    #Suma de bias
    y += bias
    
    return y


# %%
# Create a simple NN Regression Network
class RegressionModel():
    def __init__(self, input_layer_dim: int = n_features, hidden_layers: List = [3,3], output_layer = 1):
        self.input_layer = input_layer_dim
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        #Inicializacion de capas
        #Capas de entrada
        self.input_weights = np.random.rand(n_features, self.hidden_layers[0])
        self.input_bias = np.random.rand(self.hidden_layers[0])

        #Capas ocultas
        self.hidden_weights = []
        self.hidden_bias = []

        for i in range(1, hidden_layers):
            self.hidden_weights.append(np.random.randn(self.hidden_layers[i-1], self.hidden_layers[i]))
            self.hidden_bias.append()


        #Capas de salida

    def forward(self):
        pass


    def loss(self):
        pass

# %%
