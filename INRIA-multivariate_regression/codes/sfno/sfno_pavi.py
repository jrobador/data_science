#%%
import os
import torch
from data_processing import dataset_preprocessing, data_with_mask
from sph_functions import interpolation_to_grid, hemisphere_to_spherical, spherical_to_hemisphere
from plots_pavi import plot_sphere

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from model import Autoencoder, initialize_model, train_autoencoder, extract_latent_space
from metrics import test_autoencoder

DIR = r'C:\Github\pavi_data'

path_file   = os.path.join(DIR, '12_mean_sample_post.pt')
path_jobs   = os.path.join(DIR, 'jobs.json')
path_scores = os.path.join(DIR, 'scores_camcan.csv')
n_job = 44


net_number = None
sh_orders = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X, y, seed = dataset_preprocessing(DEVICE, path_file, path_jobs, path_scores)

vertices_left, vertices_right, network_left, network_right = data_with_mask(X, net_number)


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
n_scores = len(fancy_categories)

#%%
mesh_theta, sphere_src_left, sphere_src_right, sphere_dst = interpolation_to_grid(vertices_left, vertices_right, sh_orders)
sph_data_left  = hemisphere_to_spherical(network_left, sphere_src_left, sphere_dst, sh_orders)
sph_data_right = hemisphere_to_spherical(network_right, sphere_src_right, sphere_dst, sh_orders)

print(sph_data_left.shape)
print(sph_data_right.shape)

sph_data_left = sph_data_left.reshape(-1,mesh_theta.shape[0], mesh_theta.shape[1],network_left.shape[-1])
sph_data_right = sph_data_right.reshape(-1,mesh_theta.shape[0], mesh_theta.shape[1],network_right.shape[-1])

if isinstance(net_number, int):
    plot_sphere(sph_data_left, net_number)
    plot_sphere(sph_data_right, net_number)

print("New shape:")
print(sph_data_left.shape)
print(sph_data_right.shape)
#%%
autoencoders_left  = [Autoencoder(127, 254, kernel=30).to(DEVICE) for _ in range(sph_data_left.shape[-1])]
autoencoders_right = [Autoencoder(127, 254, kernel=30).to(DEVICE) for _ in range(sph_data_left.shape[-1])]

for ae_l, ae_r in zip(autoencoders_left, autoencoders_right):
    initialize_model(ae_l)
    initialize_model(ae_r)

train_data_left, test_data_left = train_test_split(sph_data_left, test_size=0.2, random_state=42)
train_data_right, test_data_right = train_test_split(sph_data_right, test_size=0.2, random_state=42)
#%%
train_latent_space_left = np.zeros((train_data_left.shape[0], 32, 64, train_data_left.shape[-1]), dtype=np.float32)
train_latent_space_right = np.zeros((train_data_right.shape[0], 32, 64, train_data_left.shape[-1]), dtype=np.float32)


for i in range (train_data_left.shape[-1]):
    print (f"Network number {1+i}")
    train_autoencoder(autoencoders_left[i],  train_data_left[:,:,:,i], DEVICE, num_iterations=100)
    train_autoencoder(autoencoders_right[i], train_data_right[:,:,:,i], DEVICE, num_iterations=100)

    lsp_left  = extract_latent_space(autoencoders_left[i], train_data_left[:,:,:,i], DEVICE)
    lsp_right = extract_latent_space(autoencoders_right[i],train_data_right[:,:,:,i], DEVICE)

    train_latent_space_left[:, :, :, i]  = lsp_left
    train_latent_space_right[:, :, :, i] = lsp_right

#%%
tr_lsp_lft = train_latent_space_left.reshape(train_latent_space_left.shape[0], -1)
tr_lsp_rght = train_latent_space_right.reshape(train_latent_space_right.shape[0], -1)

tr_lsp = np.concatenate ((tr_lsp_lft, tr_lsp_rght), axis=1)

print (tr_lsp_lft.shape)
print (tr_lsp_rght.shape)
print (tr_lsp.shape)
# %%
# Falta el test. Falta la regresion final.