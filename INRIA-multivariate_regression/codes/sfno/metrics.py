from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def metrics_spherical_rep(network, data_original_space):
    mse = mean_squared_error(network, data_original_space)
    mae = mean_absolute_error(network, data_original_space)
    Pearson_correlation = np.corrcoef(network.flatten(), data_original_space.flatten())[0, 1]

    print(f"{mse= }")
    print(f"{mae= }")
    print(f"{Pearson_correlation= }")
    print ("*"*20)