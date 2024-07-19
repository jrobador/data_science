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
import torch_harmonics
import nilearn.plotting as plotting
import hcp_utils as hcp


DIR = "/home/mind/jrobador/theta_s_dataset/plots/one_net"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (DEVICE)

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
mask = hcp.cortex_data(np.ones(len(vertices))).astype(bool)

print(f"{hcp.mesh['sphere'][0].shape=}")

#%%
vertices_left  = vertices[:32492]
vertices_right = vertices[32492:]

print (f"{vertices_left.shape=}, {vertices_right.shape=}")
#%% Representation of the network for one subject before transformation
sj_1 = data[0, :, 0]

fig_1 = plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(sj_1, fill=0), symmetric_cmap=False, cmap='Oranges',
    threshold=0.001)

file_name = "cortex_original.html"
fig_1.save_as_html(os.path.join(DIR,file_name))
plt.close()
#%%
data_cortex = np.zeros((data.shape[0], len(mask), data.shape[-1]))
data_cortex[:, mask, :] = data


print (f"{data_cortex.shape=}")

#%%
# Interpolation to a grid
sh_order = 12
nlat = int(np.sqrt(len(vertices_left)/2))
nlon = 2 * nlat
sphere_src = Sphere(xyz=vertices_left)
mesh_theta, mesh_phi = compute_new_angles_grid(nlat, nlon)
sphere_dst = Sphere(theta=mesh_theta.ravel(), phi=mesh_phi.ravel())
#%%
#TEST FOR NETWORK NUMBER 1:
net_number = 0

network_left = data_cortex[:, :len(vertices_left), net_number]
network_right= data_cortex[:, len(vertices_right):, net_number]
print (f"{network_left.shape=}, {network_right.shape=}")

sh = sf_to_sh(network_left, sphere_src, sh_order_max=sh_order)
new_data_left = sh_to_sf(sh, sphere_dst, sh_order_max=sh_order)
sh_2 = sf_to_sh(new_data_left, sphere_dst, sh_order_max=sh_order)
new_data_original_space_left = sh_to_sf(sh_2, sphere_src, sh_order_max=sh_order)

sh = sf_to_sh(network_right, sphere_src, sh_order_max=sh_order)
new_data_right = sh_to_sf(sh, sphere_dst, sh_order_max=sh_order)
sh_2 = sf_to_sh(new_data_right, sphere_dst, sh_order_max=sh_order)
new_data_original_space_right = sh_to_sf(sh_2, sphere_src, sh_order_max=sh_order)

new_data_left = new_data_left.reshape(-1, *mesh_theta.shape)
new_data_right = new_data_right.reshape(-1, *mesh_theta.shape)
print (f"{new_data_right.shape=}")
print (f"{new_data_left.shape =}")


fig = plt.figure(layout='constrained')
fig.suptitle("Resampled data for network " + str(net_number+1) + " left")
subfigs = fig.subfigures(1, 5).ravel()

for i, subfig in enumerate(subfigs):
    plots.plot_sphere(new_data_left[i], fig=subfig, cmap='plasma')

fig = plt.figure(layout='constrained')
fig.suptitle("Resampled data for network " + str(net_number+1) + " right")
subfigs = fig.subfigures(1, 5).ravel()

for i, subfig in enumerate(subfigs):
   plots.plot_sphere(new_data_right[i], fig=subfig, cmap='plasma')

#%%
print(new_data_original_space_left.shape)
print(new_data_left.shape)
print(new_data_right.shape)
#%%
fig_2 = plotting.view_surf(hcp.mesh.inflated, np.hstack([new_data_original_space_left[0],new_data_original_space_right[0]]).clip(0, 1), symmetric_cmap=False, cmap='Oranges',
    threshold=0.001)

file_name = "c_four=" + str(sh_order)+ "_net=1.html"
fig_2.save_as_html(os.path.join(DIR,file_name))
plt.close()
#%%
# Autoencoder architecture which reduces dimensionality to half

kernel = 30
vmfconvolution_learnable_left = gnn.Sequential(
    'x0',
    [
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, output_ratio=.5, weights=False, bias=True), 'x0->x1'),
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.5, output_ratio=.25, weights=False, bias=True), 'x1->x2'),
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.25, output_ratio=.5, weights=False, bias=True), 'x2->x3 '),
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.5, weights=False, bias=True), 'x3 + x1->x4')
    ]
).to(DEVICE)

vmfconvolution_learnable_right = gnn.Sequential(
    'x0',
    [
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, output_ratio=.5, weights=False, bias=True), 'x0->x1'),
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.5, output_ratio=.25, weights=False, bias=True), 'x1->x2'),
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.25, output_ratio=.5, weights=False, bias=True), 'x2->x3 '),
        (vmf_convolution.VMFConvolution(kernel, nlat, nlon, input_ratio=.5, weights=False, bias=True), 'x3 + x1->x4')
    ]
).to(DEVICE)

# Autoencoder training
# Initialize close to identity
for step in vmfconvolution_learnable_left:
    nn.init.ones_(step.bias)

# Initialize close to identity
for step in vmfconvolution_learnable_right:
    nn.init.ones_(step.bias)

lr = 1.
optim_left = torch.optim.SGD(vmfconvolution_learnable_left.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim_left, 50, gamma=0.1)

new_data_tensor_left = torch.Tensor(np.exp(-new_data_left)).to(DEVICE)
new_data_tensor_left_normed = nn.functional.normalize(torch.exp(-new_data_tensor_left), dim=[-2, -1])

losses = []
for iter in range(1001):
    optim_left.zero_grad(set_to_none=True)

    # Forward pass
    prediction = nn.functional.normalize(vmfconvolution_learnable_left(new_data_tensor_left_normed), dim=[-2, -1])

    # Compute loss
    loss = ((prediction - new_data_tensor_left_normed) ** 2).sum(dim=[-2, -1]).mean()

    # Backward pass
    loss.backward()

    # Optimization step
    optim_left.step()

    # Append and print loss
    loss_value = loss.item()
    losses.append(loss_value)
    if (iter % 50 == 0):
            print(f"Iteration {iter}, Loss: {loss_value}")

with torch.no_grad():
    predicted_left = vmfconvolution_learnable_left(new_data_tensor_left_normed)

fig = plt.figure(layout='constrained')
subfigs = fig.subfigures(2, 5)
fig.suptitle(f"Real data vs autoencoder output. Left \n Kernel = " + str(kernel))

for i in range(min(subfigs.shape[1], predicted_left.shape[0])):
    plots.plot_sphere(new_data_tensor_left_normed[i].cpu().numpy(), cmap='plasma', fig=subfigs[0, i])
    plots.plot_sphere(predicted_left[i].cpu().numpy(), cmap='plasma', fig=subfigs[1, i])

# Print learned parameters
with torch.no_grad():
    x_ = vmfconvolution_learnable_left[0].kernel()
    x__ = vmfconvolution_learnable_left[-1].kernel()

plt.figure(layout='constrained')
plots.plot_sphere(x_.cpu(), colorbar=True, vmin=-5, vmax=5, central_latitude=90, title=f"Learned parameters (first layer) \n Kernel = " + str(kernel))
plt.figure(layout='constrained')
plots.plot_sphere(x__.cpu(), colorbar=True, vmin=-5, vmax=5, central_latitude=90, title=f"Learned parameters (last layer) \n Kernel = " + str(kernel))


# %%
lr = 1.
optim_right = torch.optim.SGD(vmfconvolution_learnable_right.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim_right, 50, gamma=0.1)

new_data_tensor_right = torch.Tensor(np.exp(-new_data_right)).to(DEVICE)
new_data_tensor_right_normed = nn.functional.normalize(torch.exp(-new_data_tensor_right), dim=[-2, -1])

losses = []
for iter in range(1001):
    optim_right.zero_grad(set_to_none=True)

    # Forward pass
    prediction = nn.functional.normalize(vmfconvolution_learnable_right(new_data_tensor_right_normed), dim=[-2, -1])

    # Compute loss
    loss = ((prediction - new_data_tensor_right_normed) ** 2).sum(dim=[-2, -1]).mean()

    # Backward pass
    loss.backward()

    # Optimization step
    optim_left.step()

    # Append and print loss
    loss_value = loss.item()
    losses.append(loss_value)
    if (iter % 50 == 0):
            print(f"Iteration {iter}, Loss: {loss_value}")

with torch.no_grad():
    predicted_right = vmfconvolution_learnable_right(new_data_tensor_right_normed)

fig = plt.figure(layout='constrained')
subfigs = fig.subfigures(2, 5)
fig.suptitle(f"Real data vs autoencoder output. Right \n Kernel = " + str(kernel))

for i in range(min(subfigs.shape[1], predicted_right.shape[0])):
    plots.plot_sphere(new_data_tensor_right_normed[i].cpu().numpy(), cmap='plasma', fig=subfigs[0, i])
    plots.plot_sphere(predicted_right[i].cpu().numpy(), cmap='plasma', fig=subfigs[1, i])

# Print learned parameters
with torch.no_grad():
    x_ = vmfconvolution_learnable_right[0].kernel()
    x__ = vmfconvolution_learnable_right[-1].kernel()

plt.figure(layout='constrained')
plots.plot_sphere(x_.cpu(), colorbar=True, vmin=-5, vmax=5, central_latitude=90, title=f"Learned parameters (first layer) \n Kernel = " + str(kernel))
plt.figure(layout='constrained')
plots.plot_sphere(x__.cpu(), colorbar=True, vmin=-5, vmax=5, central_latitude=90, title=f"Learned parameters (last layer) \n Kernel = " + str(kernel))

#%% Reconstructed after convolution

predicted_reshaped_left = predicted_left.reshape(predicted_left.shape[0], np.prod(predicted_left.shape[1:]))
print ("Autoencoder output left reshaped to:")
print (predicted_reshaped_left.shape)

sh_inverse = sf_to_sh(predicted_reshaped_left, sphere_dst, sh_order_max=20)
print(sh_inverse.shape)

new_data_inverse_left = sh_to_sf(sh_inverse, sphere_src, sh_order_max=20)
print (new_data_inverse_left.shape)


predicted_reshaped_right = predicted_right.reshape(predicted_right.shape[0], np.prod(predicted_right.shape[1:]))
print ("Autoencoder output right reshaped to:")
print (predicted_reshaped_right.shape)

sh_inverse = sf_to_sh(predicted_reshaped_right, sphere_dst, sh_order_max=20)
print(sh_inverse.shape)

new_data_inverse_right = sh_to_sf(sh_inverse, sphere_src, sh_order_max=20)
print (new_data_inverse_right.shape)
# %%
# Plotting the inverse

nd_inverse_one_subject_left = new_data_inverse_left[0,:]
nd_inverse_one_subject_right = new_data_inverse_right[0,:]


print (nd_inverse_one_subject_left.shape)
print (nd_inverse_one_subject_right.shape)

# %%
fig_3 = plotting.view_surf(hcp.mesh.inflated, np.hstack([nd_inverse_one_subject_left, nd_inverse_one_subject_right]).clip(0, 1), symmetric_cmap=False, cmap='Oranges',
    threshold=0.001)
# %%
fig_3
# %%
fig_2
# %%
fig_1
# %%
