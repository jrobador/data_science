import torch.nn as nn
import torch
from torch_geometric import nn as gnn
import vmf_convolution
import numpy as np

class Encoder(nn.Module):
    def __init__(self, nlat, nlon, kernel=30):
        super(Encoder, self).__init__()
        self.encoder = gnn.Sequential(
            'x0',
            [
                (vmf_convolution.VMFConvolution(kernel, nlat, nlon, output_ratio=.5, weights=False, bias=True), 'x0->x1'),
                (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.5, output_ratio=.25, weights=False, bias=True), 'x1->x2')
            ]
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, nlat, nlon, kernel=30):
        super(Decoder, self).__init__()
        self.decoder = gnn.Sequential(
            'x0',
            [
                (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.25, output_ratio=.5, weights=False, bias=True), 'x0->x1'),
                (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.5, weights=False, bias=True), 'x1->x2')
            ]
        )
    
    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, nlat, nlon, kernel=30):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(nlat, nlon, kernel)
        self.decoder = Decoder(nlat, nlon, kernel)
    
    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed, latent_space
    

def initialize_model(model):
    for layer in model.parameters():
        if layer.requires_grad:
            nn.init.ones_(layer)


def train_autoencoder(model, data, DEVICE, num_iterations=2000, lr=1.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    data_tensor = torch.Tensor(np.exp(-data)).to(DEVICE)
    data_tensor_normed = nn.functional.normalize(torch.exp(-data_tensor), dim=[-2, -1])
    
    losses = []
    for iter in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        reconstructed, _ = model(data_tensor_normed)

        reconstructed_normed = nn.functional.normalize(reconstructed, dim=[-2, -1])
        
        loss = ((reconstructed_normed - data_tensor_normed) ** 2).sum(dim=[-2, -1]).mean()
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        # Append and print loss
        loss_value = loss.item()
        losses.append(loss_value)
        if iter % 50 == 0:
            print(f"Iteration {iter}, Loss: {loss_value}")
    
    return losses


def extract_latent_space(model, data, DEVICE):
    data_tensor = torch.Tensor(np.exp(-data)).to(DEVICE)
    data_tensor_normed = nn.functional.normalize(torch.exp(-data_tensor), dim=[-2, -1])
    
    with torch.no_grad():
        _, latent_space = model(data_tensor_normed)
    
    return latent_space.cpu().numpy()