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

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import json

import pandas as pd

DIR = r'C:\Github\data_science\INRIA-multivariate_regression\codes\sfno\figures'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (DEVICE)

def compute_new_angles_grid(nlat, nlon):
    new_phi = np.linspace(0, 2 * np.pi, nlon + 1)[1:]
    new_theta = np.linspace(0, np.pi, nlat)
    new_phi, new_theta = np.meshgrid(new_phi, new_theta)

    return new_theta, new_phi

# Data and vertices representation

file_path = os.path.join(r'C:\Github\pavi_data', '12_mean_sample_post.pt')
path_scores = os.path.join(r'C:\Github\pavi_data', 'scores_camcan.csv')
path_jobs = os.path.join(r'C:\Github\pavi_data\jobs.json')

sample = torch.load(file_path, map_location=DEVICE)

temperature = 1

## Para la regresion no se necesita el softmax!
#data = softmax(sample['theta_s'].detach().cpu() / temperature, axis=-1)
data = sample['theta_s'].detach().cpu()

#%%
# Handling missing values

categories = [
    'BentonFaces_total',
    'Cattell_total',
    'EkmanEmHex_pca1',
    'FamousFaces_details',
    'Hotel_time',
    'PicturePriming_baseline_acc',
    'Proverbs',
    'Synsem_prop_error',
    'Synsem_RT',
    'VSTMcolour_K_mean',
    'VSTMcolour_K_precision',
    'VSTMcolour_K_doubt',
    'VSTMcolour_MSE'
]

fancy_categories = [
    'Benton faces',
    'Fluid Intelligence',
    'Emotion expression recognition',
    'Famous faces',
    'Hotel task',
    'Picture priming',
    'Proverb comprehension',
    'Sentence comprehension (unacceptable error)',
    'Sentence comprehension (reaction time)',
    'Visual short term memory (mean)',
    'Visual short term memory (precision)',
    'Visual short term memory (doubt)',
    'Visual short term memory (MSE)',
]

n_job = 44

with open(path_jobs, 'r') as f:
    jobs = json.load(f)
seed = jobs['seed'][n_job]
subjects_list = jobs['sub_list'][n_job]

subjects_list = np.array([str(s) for s in subjects_list])

def _preprocess_scores(data):
    data.Subject = data.Subject.astype('str')
    
    # Discard subjects with Nan-valued scores or subjects not in the list
    mask = data[categories].isna().sum(axis=1) == 0
    id_subjects_ok = set(data[mask].Subject) & set(subjects_list)
    mask = data.Subject.isin(id_subjects_ok)
    data = data[mask][categories + ['Subject']]
    data = data.set_index('Subject')
    scaler = StandardScaler()
    data = pd.DataFrame(
        data=scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    mask = pd.Series(subjects_list).isin(id_subjects_ok).to_numpy()
    return data.loc[subjects_list[mask]], mask

scores = pd.read_csv(path_scores)
scores, scores_mask = _preprocess_scores(scores)

y = scores.to_numpy()
X = np.array([x.cpu().numpy() for x in data])[scores_mask]
data = X

vertices = hcp.mesh['sphere'][0] / np.linalg.norm(hcp.mesh['sphere'][0], axis=1, keepdims=True)
mask = hcp.cortex_data(np.ones(len(vertices))).astype(bool)
print(f"{hcp.mesh['sphere'][0].shape=}")

#%%
print (X.shape)
print (data.shape)

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
sh_order = 48
print (f"{sh_order=}")
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

mse = mean_squared_error(network_left, new_data_original_space_left)
mae = mean_absolute_error(network_left, new_data_original_space_left)
Pearson_correlation = np.corrcoef(network_left.flatten(), new_data_original_space_left.flatten())[0, 1]
print ("For left side:")
print(f"{mse= }")
print(f"{mae= }")
print(f"{Pearson_correlation= }")
print ("*"*20)


sh = sf_to_sh(network_right, sphere_src, sh_order_max=sh_order)
new_data_right = sh_to_sf(sh, sphere_dst, sh_order_max=sh_order)
sh_2 = sf_to_sh(new_data_right, sphere_dst, sh_order_max=sh_order)
new_data_original_space_right = sh_to_sf(sh_2, sphere_src, sh_order_max=sh_order)

mse = mean_squared_error(network_right, new_data_original_space_right)
mae = mean_absolute_error(network_right, new_data_original_space_right)
Pearson_correlation = np.corrcoef(network_right.flatten(), new_data_original_space_right.flatten())[0, 1]
print ("For right side:")
print(f"{mse= }")
print(f"{mae= }")
print(f"{Pearson_correlation= }")


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
def compute_corrected_explained_variance_ratio(sh_coeffs, sh_order_max):
    variances = []
    
    total_variance = np.sum(np.nanvar(sh_coeffs, axis=0))
    
    for order in range(sh_order_max + 1):
        # Extract coefficients corresponding to this order
        order_coeffs = extract_coeffs_for_order(sh_coeffs, order)
        
        if len(order_coeffs) > 0 and not np.isnan(order_coeffs).all() and not np.isinf(order_coeffs).all():
            order_variance = np.sum(np.nanvar(order_coeffs, axis=0))
        else:
            order_variance = 0
        
        variances.append(order_variance)
    
    explained_variance_ratio = np.array(variances) / total_variance
    return explained_variance_ratio

def extract_coeffs_for_order(sh_coeffs, order):
    start_idx = order ** 2
    end_idx = (order + 1) ** 2
    return sh_coeffs[:, start_idx:end_idx]

def plot_explained_variance_ratio(explained_variance_ratio):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(explained_variance_ratio)), explained_variance_ratio, linestyle='-', color='b')
    plt.title('Explained Variance Ratio by SH Order')
    plt.xlabel('SH Order')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    file_name = "expl_var.png"
    plt.savefig(os.path.join(DIR,file_name))
    plt.show()

explained_variance_ratio = compute_corrected_explained_variance_ratio(sh, sh_order)
print("Sum of Explained Variance Ratios:", np.sum(explained_variance_ratio))

plot_explained_variance_ratio(explained_variance_ratio)

#%%
fig_2 = plotting.view_surf(hcp.mesh.inflated, np.hstack([new_data_original_space_left[0],new_data_original_space_right[0]]).clip(0, 1), symmetric_cmap=False, cmap='Oranges',
    threshold=0.001)

file_name = "c_four=" + str(sh_order)+ "_net=1.html"
fig_2.save_as_html(os.path.join(DIR,file_name))
plt.close()

#%%
train_data_left, test_data_left = train_test_split(new_data_left, test_size=0.2, random_state=42)
train_data_right, test_data_right = train_test_split(new_data_right, test_size=0.2, random_state=42)

#%%
# Autoencoder architecture which reduces dimensionality to half

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

def train_autoencoder(model, data, num_iterations=1000, lr=1.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    data_tensor = torch.Tensor(np.exp(-data)).to(DEVICE)
    data_tensor_normed = nn.functional.normalize(torch.exp(-data_tensor), dim=[-2, -1])
    
    losses = []
    for iter in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        reconstructed, latent_space = model(data_tensor_normed)

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

def extract_latent_space(model, data):
    data_tensor = torch.Tensor(np.exp(-data)).to(DEVICE)
    data_tensor_normed = nn.functional.normalize(torch.exp(-data_tensor), dim=[-2, -1])
    
    with torch.no_grad():
        _, latent_space = model(data_tensor_normed)
    
    return latent_space.cpu().numpy()

def test_model(model, test_data):
    #Calculo de metricas!!!
    test_data_tensor = torch.Tensor(np.exp(-test_data)).to(DEVICE)
    test_data_tensor_normed = nn.functional.normalize(torch.exp(-test_data_tensor), dim=[-2, -1])
    
    with torch.no_grad():
        predicted, _ = model(test_data_tensor_normed)
    
    return test_data_tensor_normed, predicted

#Calculo de Varianza explicada por cada componente!!!
def explained_variance_autoencoder(model, test_data):
    pass

def plot_results(real_data, predicted_data, kernel, side, model):
    fig = plt.figure(layout='constrained')
    subfigs = fig.subfigures(2, 5)
    fig.suptitle(f"Real data vs autoencoder output. {side} \n Kernel = {kernel}")
    
    for i in range(min(subfigs.shape[1], predicted_data.shape[0])):
        plots.plot_sphere(real_data[i].cpu().numpy(), cmap='plasma', fig=subfigs[0, i])
        plots.plot_sphere(predicted_data[i].cpu().numpy(), cmap='plasma', fig=subfigs[1, i])
    
    # Print learned parameters
    with torch.no_grad():
        first_layer_kernel = model[0].kernel()
        last_layer_kernel = model[-1].kernel()
    
    plt.figure(layout='constrained')
    plots.plot_sphere(first_layer_kernel.cpu(), colorbar=True, vmin=-5, vmax=5, central_latitude=90, title=f"Learned parameters (first layer) \n Kernel = {kernel}")
    
    plt.figure(layout='constrained')
    plots.plot_sphere(last_layer_kernel.cpu(), colorbar=True, vmin=-5, vmax=5, central_latitude=90, title=f"Learned parameters (last layer) \n Kernel = {kernel}")

autoencoder_left  = Autoencoder(nlat, nlon, kernel=30).to(DEVICE)
autoencoder_right = Autoencoder(nlat, nlon, kernel=30).to(DEVICE)

initialize_model(autoencoder_left)
initialize_model(autoencoder_right)

train_data_left, test_data_left = train_test_split(new_data_left, test_size=0.2, random_state=42)
train_data_right, test_data_right = train_test_split(new_data_right, test_size=0.2, random_state=42)

losses_left = train_autoencoder(autoencoder_left, train_data_left)
losses_right = train_autoencoder(autoencoder_right, train_data_right)

latent_train_left = extract_latent_space(autoencoder_left, train_data_left)
latent_test_left = extract_latent_space(autoencoder_left, test_data_left)

latent_train_right = extract_latent_space(autoencoder_right, train_data_right)
latent_test_right = extract_latent_space(autoencoder_right, test_data_right)

test_data_tensor_left_normed, predicted_left = test_model(autoencoder_left, test_data_left)
test_data_tensor_right_normed, predicted_right = test_model(autoencoder_right, test_data_right)

plot_results(test_data_tensor_left_normed, predicted_left, kernel=30, side='Left', model=autoencoder_left)
plot_results(test_data_tensor_right_normed, predicted_right, kernel=30, side='Right', model=autoencoder_right)

# %% 
# Reconstructed after convolution
predicted_reshaped_left = predicted_left.reshape(predicted_left.shape[0], np.prod(predicted_left.shape[1:]))
print ("Autoencoder output left reshaped to:")
print (predicted_reshaped_left.shape)

sh_inverse = sf_to_sh(predicted_reshaped_left.cpu(), sphere_dst, sh_order_max=sh_order)
new_data_inverse_left = sh_to_sf(sh_inverse, sphere_src, sh_order_max=sh_order)


predicted_reshaped_right = predicted_right.reshape(predicted_right.shape[0], np.prod(predicted_right.shape[1:]))
print ("Autoencoder output right reshaped to:")
print (predicted_reshaped_right.shape)

sh_inverse = sf_to_sh(predicted_reshaped_right.cpu(), sphere_dst, sh_order_max=sh_order)
new_data_inverse_right = sh_to_sf(sh_inverse, sphere_src, sh_order_max=sh_order)
# %%
# Plotting the inverse

nd_inverse_one_subject_left = new_data_inverse_left[0,:]
nd_inverse_one_subject_right = new_data_inverse_right[0,:]
fig_3 = plotting.view_surf(hcp.mesh.inflated, np.hstack([nd_inverse_one_subject_left, nd_inverse_one_subject_right]).clip(0, 1), symmetric_cmap=False, cmap='Oranges',
    threshold=0.001)

file_name = "autoenc_output.html"
fig_3.save_as_html(os.path.join(DIR,file_name))
plt.close()

# %%
fig_1
# %%
fig_2
# %%
fig_3
# %%
