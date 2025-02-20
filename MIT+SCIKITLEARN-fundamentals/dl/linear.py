import numpy as np

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


#%%
import numpy as np
y_a = np.array([[1,1,1,1,1],[3,1,4,2,1],[1,2,3,4,5]])
bias = np.array([1,2,3,4,5])
# %%
y_a.shape #(batch_size, output_dim)

# %%
bias.shape
# %%
y = y_a + bias
print(y)
# %%
y_a
# %%
