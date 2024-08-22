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

DIR = r'C:\Github\pavi_data'

path_file   = os.path.join(DIR, '12_mean_sample_post.pt')
path_jobs   = os.path.join(DIR, 'jobs.json')
path_scores = os.path.join(DIR, 'scores_camcan.csv')
n_job = 44


net_number = None
sh_orders = [80]

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
mean_scores = []
for sh_order in sh_orders:
    mesh_theta, sphere_src_left, sphere_src_right, sphere_dst = interpolation_to_grid(vertices_left, vertices_right, sh_order)
    sph_data_left  = hemisphere_to_spherical(network_left, sphere_src_left, sphere_dst, sh_order)
    sph_data_right = hemisphere_to_spherical(network_right, sphere_src_right, sphere_dst, sh_order)

    #Me parece que podriamos hacerlo sin reshapearlo a la mesh...
    #sph_data_left = sph_data_left.reshape(-1, *mesh_theta.shape)
    #sph_data_right = sph_data_right.reshape(-1, *mesh_theta.shape)

    if isinstance(net_number, int):
        sph_data_left  = sph_data_left.reshape(-1, *mesh_theta.shape)
        sph_data_right = sph_data_right.reshape(-1, *mesh_theta.shape)
        plot_sphere(sph_data_left, net_number)
        plot_sphere(sph_data_right, net_number)
    

    sph_data =  np.concatenate((sph_data_left, sph_data_right), axis=1)
    print (sph_data.shape)
    sph_data = sph_data.reshape(sph_data_left.shape[0], -1)
    print (sph_data.shape)

    X_train, X_test, y_train, y_test = train_test_split(sph_data, y, random_state=42)

    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores_ = np.full((n_scores,), np.nan)
    for c in range(n_scores):
        r, _ = pearsonr(y_pred[:, c], y_test[:, c])
        scores_[c] = r

    mean_scores.append(scores_.mean())

#%%
plt.figure(figsize=(10,8))
plt.plot(sh_orders, mean_scores, linestyle='-', color='b', marker='o')
plt.title('r-Pearson correlation by SH Order')
plt.xlabel('SH Order')
plt.ylabel('r-Pearson correlation')
plt.grid(True)
plt.savefig(os.path.join(DIR,'p-score_vs_sh-order2.png'))
plt.show()
# %%
scores_
# %%
mean_scores
# %%

def plot_k_fold(predicted_scores, n_splits):
    df_pred = pd.DataFrame(predicted_scores, columns=fancy_categories)
    df_pred = df_pred.reset_index()
    df_pred['index'] = df_pred['index'] // n_splits
    df_pred = df_pred.groupby('index').mean()


    mean = y.to_numpy().mean()
    std = y.to_numpy().mean(axis=1).std()
    fig, ax = plt.subplots()
    sns.boxplot(
        data=y, width=0.3, color='skyblue', ax=ax, showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "5"
        }
    )
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='grey', linestyle='dashed')
    # plt.ylim(-0.5, 0.5)
    plt.ylabel('r-Pearson correlation')
    title = "Regression with Spherical Harmonics"
    plt.title(title + f"\n Mean score: {mean:.4f} +/- {std:.4f}")
    plt.tight_layout()
    name_file = f"r-pearson_k-folds={n_splits}"
    plt.savefig(os.path.join(DIR, name_file + '.png'))
    plt.close()