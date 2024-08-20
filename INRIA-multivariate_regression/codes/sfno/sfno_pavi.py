#%%
import os
import torch
from data_processing import dataset_preprocessing, data_with_mask
from sph_functions import interpolation_to_grid, hemisphere_to_spherical, spherical_to_hemisphere
from plots_pavi import plot_sphere
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import pearsonr


file_path = os.path.join(r'C:\Github\pavi_data', '12_mean_sample_post.pt')
path_jobs = os.path.join(r'C:\Github\pavi_data\jobs.json')
path_scores = os.path.join(r'C:\Github\pavi_data', 'scores_camcan.csv')
n_job = 44

net_number = None
sh_order = 24

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X, y, seed = dataset_preprocessing(DEVICE, file_path, path_jobs, path_scores)

vertices_left, vertices_right, network_left, network_right = data_with_mask(X, net_number)

mesh_theta, sphere_src_left, sphere_src_right, sphere_dst = interpolation_to_grid(vertices_left, vertices_right, sh_order)
#%%
sph_data_left  = hemisphere_to_spherical(network_left, sphere_src_left, sphere_dst, sh_order)
sph_data_right = hemisphere_to_spherical(network_right, sphere_src_right, sphere_dst, sh_order)

#Me parece que podriamos hacerlo sin reshapearlo a la mesh...
#sph_data_left = sph_data_left.reshape(-1, *mesh_theta.shape)
#sph_data_right = sph_data_right.reshape(-1, *mesh_theta.shape)

if isinstance(net_number, int):
    sph_data_left = sph_data_left.reshape(-1, *mesh_theta.shape)
    sph_data_right = sph_data_right.reshape(-1, *mesh_theta.shape)
    plot_sphere(sph_data_left, net_number)
    plot_sphere(sph_data_right, net_number)

# %%
import numpy as np

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
sph_data =  np.concatenate((sph_data_left, sph_data_right), axis=1)
print (sph_data.shape)
sph_data = sph_data.reshape(218, -1)
print (sph_data.shape)

#%%
sph_data = sph_data.astype(np.float32)
y = y.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(sph_data, y, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(1000,500),random_state=42, max_iter=2500, verbose=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

scores_ = np.full((n_scores,), np.nan)
for c in range(n_scores):
    r, _ = pearsonr(y_pred[:, c], y_test[:, c])
    scores_[c] = r

print (scores_)
#%%
y.shape

# %%
X.shape
# %%
y_pred
y_test
# %%
y.std()
# %%
sph_data.mean()
# %%
sph_data.std()
