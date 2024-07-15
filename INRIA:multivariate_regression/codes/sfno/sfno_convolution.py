# %%
import pickle
from pathlib import Path
from importlib import reload

from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.core.sphere import Sphere

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
try:
    import vmf_convolution
except ImportError:
    import vmf_convolution
reload(vmf_convolution)
import plots
from torch_geometric import nn as gnn

from scipy.special import softmax

import os

import hcp_utils as hcp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_new_angles_grid(nlat, nlon):
    new_phi = np.linspace(0, 2 * np.pi, nlon + 1)[1:]
    new_theta = np.linspace(0, np.pi, nlat)
    new_phi, new_theta = np.meshgrid(new_phi, new_theta)

    return new_theta, new_phi

# Data and vertices representation

file_path = os.path.join('/home/mind/jrobador/3_camcan', f'12_mean_sample_post.pt')
sample = torch.load(file_path, map_location=DEVICE)

temperature = 1
data = softmax(sample['theta_s'].detach().cpu() / temperature, axis=-1)

vertices = hcp.mesh['sphere'][0] / np.linalg.norm(hcp.mesh['sphere'][0], axis=1, keepdims=True)
mask = hcp.cortex_data(np.ones(59412)).astype(bool)

vertices = vertices[mask]

### Testing for the first network here!
nbr = 0
first_network = data[:,:,nbr]

# Interpolation to a grid

nlat = 16
nlon = 2 * nlat

sphere_src = Sphere(xyz=vertices)
mesh_theta, mesh_phi = compute_new_angles_grid(nlat, nlon)
sphere_dst = Sphere(theta=mesh_theta.ravel(), phi=mesh_phi.ravel())
sh = sf_to_sh(first_network, sphere_src, sh_order_max=4)
new_data = sh_to_sf(sh, sphere_dst, sh_order_max=4).reshape(-1, *mesh_theta.shape)

fig = plt.figure(layout='constrained')
fig.suptitle("Resampled Data for network " + str(nbr+1))
subfigs = fig.subfigures(1, 5).ravel()
for i, subfig in enumerate(subfigs):
    plots.plot_sphere(new_data[i], fig=subfig, cmap='plasma')

# Showing convolutions

fig = plt.figure(layout='constrained')
fig.suptitle("Resampled Data $e^{-x}$ for network " + str(nbr+1))
subfigs = fig.subfigures(1, 5).ravel()
for i, subfig in enumerate(subfigs):
    plots.plot_sphere(
        np.exp(-new_data)[i],
        #title='before convolution',
        cmap='plasma',
        fig=subfig
    )

new_data_tensor = torch.Tensor(np.exp(-new_data)).to(DEVICE)
for kappa in (1, 5, 30):
    vmfconvolution = vmf_convolution.VMFConvolution(kappa, nlat, nlon, output_ratio=.5).to(DEVICE)
    new_data_conv = vmfconvolution(new_data_tensor)

    plots.plot_sphere(vmfconvolution.kernel().cpu().numpy(), cmap='plasma', title=f'kernel $\\kappa={kappa}$')

    fig = plt.figure(layout='constrained')
    fig.suptitle(f"$e^{{-x}}$ convolved with kappa={kappa}")
    subfigs = fig.subfigures(1, 5).ravel()
    for i, subfig in enumerate(subfigs):
        plots.plot_sphere(
            new_data_conv[i].cpu().numpy(),
            #title=f'after convolution $\kappa={kappa}$', 
            cmap='plasma',
            # colorbar=True,
            fig=subfig
        )

# Autoencoder which reduces dimensionality to half

vmfconvolution_learnable = gnn.Sequential(
    'x0',
    [
        (vmf_convolution.VMFConvolution(None, nlat, nlon, output_ratio=.5, weights=False, bias=True), 'x0->x1'),
        (vmf_convolution.VMFConvolution(None, nlat, nlon, input_ratio=.5, output_ratio=.25, weights=False, bias=True), 'x1->x2'),
        (vmf_convolution.VMFConvolution(None, nlat, nlon, input_ratio=.25, output_ratio=.5, weights=False, bias=True), 'x2->x3 '),
        (vmf_convolution.VMFConvolution(None, nlat, nlon, input_ratio=.5, weights=False, bias=True), 'x3 + x1->x4')
    ]
).to(DEVICE)

# Initialize close to identity
for step in vmfconvolution_learnable:
    nn.init.ones_(step.bias)

lr = 1.
optim = torch.optim.SGD(vmfconvolution_learnable.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 50, gamma=0.1)

new_data_tensor_normed = nn.functional.normalize(torch.exp(-new_data_tensor), dim=[-2, -1])
losses = []
for iter in range(1000):
    optim.zero_grad(set_to_none=True)
    prediction = nn.functional.normalize(vmfconvolution_learnable(new_data_tensor_normed), dim=[-2, -1])
    loss = ((prediction - new_data_tensor_normed) ** 2).sum(dim=[-2, -1]).mean()
    loss.backward()
    optim.step()
    losses.append(loss.item())

plt.semilogy(losses)

with torch.no_grad():
    predicted = vmfconvolution_learnable(new_data_tensor_normed)

fig = plt.figure(layout='constrained')
subfigs = fig.subfigures(2, 5)
for i in range(min(subfigs.shape[1], predicted.shape[0])):
    plots.plot_sphere(new_data_tensor_normed[i].cpu().numpy(), cmap='plasma', fig=subfigs[0, i])
    plots.plot_sphere(predicted[i].cpu().numpy(), cmap='plasma', fig=subfigs[1, i])


# Print learned parameters

with torch.no_grad():
    x_ = vmfconvolution_learnable[0].kernel()
    x__ = vmfconvolution_learnable[-1].kernel()
plots.plot_sphere(x_, colorbar=True, vmin=-5, vmax=5, central_latitude=90, title=f'Learned parameters')
plots.plot_sphere(x__, colorbar=True, vmin=-5, vmax=5, central_latitude=90)

# %%
