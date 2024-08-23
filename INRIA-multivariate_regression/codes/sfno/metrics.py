from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import torch.nn as nn

def metrics_spherical_rep(network, data_original_space):
    mse = mean_squared_error(network, data_original_space)
    mae = mean_absolute_error(network, data_original_space)
    Pearson_correlation = np.corrcoef(network.flatten(), data_original_space.flatten())[0, 1]

    print(f"{mse= }")
    print(f"{mae= }")
    print(f"{Pearson_correlation= }")
    print ("*"*20)


def test_autoencoder(model, test_data, DEVICE):
    test_data_tensor = torch.Tensor(np.exp(-test_data)).to(DEVICE)
    test_data_tensor_normed = nn.functional.normalize(torch.exp(-test_data_tensor), dim=[-2, -1])
    
    with torch.no_grad():
        predicted, _ = model(test_data_tensor_normed)

    # Convert tensors to numpy arrays for metric calculation
    test_data_np = test_data_tensor_normed.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    
    data_tensor = torch.Tensor(np.exp(-test_data)).to(DEVICE)
    data_tensor_normed = nn.functional.normalize(torch.exp(-data_tensor), dim=[-2, -1])

    reconstructed_normed = nn.functional.normalize(predicted, dim=[-2, -1])

    # Compute MAE and MSE
    mae = mean_absolute_error(test_data_np.flatten(), predicted_np.flatten())
    mse = mean_squared_error(test_data_np.flatten(), predicted_np.flatten())
    loss = ((reconstructed_normed - data_tensor_normed) ** 2).sum(dim=[-2, -1]).mean()

    # Compute correlation coefficient
    corrcoef = np.corrcoef(test_data_np.flatten(), predicted_np.flatten())[0, 1]
    
    metrics = {
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
        'correlation_coefficient': corrcoef,
        'loss': loss.item()
    }

    print (metrics)
    
    return test_data_tensor_normed, predicted