#%%
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


def linear_function(inp, weights, bias):
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

    #La salida es (batch_size, output_dim). Olvidarse un poco del concepto del batch_size
    return y


# %%
# Create a simple NN Regression Network
class RegressionModel():
    def __init__(self, input_layer_dim: int = n_features, hidden_layers: List = [3,3], output_layer_dim = 1):
        self.input_layer = input_layer_dim
        self.hidden_layers = hidden_layers
        self.output_layer_dim = output_layer_dim


    def initialization(self):
        #Inicializacion de capas
        #Capas de entrada
        self.input_weights = np.random.rand(self.input_layer, self.hidden_layers[0])
        self.input_bias = np.random.rand(self.hidden_layers[0])

        #Capas ocultas
        self.hidden_weights = []
        self.hidden_bias = []

        for i in range(1, len(self.hidden_layers)):
            self.hidden_weights.append(np.random.randn(self.hidden_layers[i-1], self.hidden_layers[i]))
            self.hidden_bias.append(np.random.rand(self.hidden_layers[i]))

        #Capas de salida
        self.output_weights = np.random.rand(self.hidden_layers[-1], self.output_layer_dim)
        self.output_bias    = np.random.rand(self.output_layer_dim)

    def forward(self, x):
        #Propagacion hacia adelante

        #Capa de entrada
        # X -> (batch_size, features)
        # input_linear -> (batch_size, hidden[0])
        input_linear = linear_function(x, self.input_weights, self.input_bias)

        #Capas ocultas
        # input_linear -> (batch_size, hidden[0])
        # hidden_linear[n] -> (hidden[0], output_size)
        linear_y = []
        input_iter = input_linear
        for i in range (1, len(self.hidden_layers)):
            linear_y.append(linear_function(input_iter, self.hidden_weights[i], self.hidden_bias[i]))
            input_iter = linear_y[-1]
        
        #Capas de salida
        output_linear = linear_function(input_iter, self.output_weights, self.output_bias)

        return output_linear

    def loss(self, y, forward_output):
        # Vamos a calcular la suma cuadrática media
        MSE = np.mean((y - forward_output)**2)

        return MSE

    def backpropagation(self):
        pass

# %%
