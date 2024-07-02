# %%
# import argparse
from typing import Optional, List, Any, Union
from pathlib import Path
import sys
import json
from time import time
from tqdm import tqdm
from typing import Dict
import argparse
import pickle

import numpy as np
from scipy.special import softmax
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

import torch
import torch.nn as nn

import os

import pyro

from script_training import main
from skorch import NeuralNetRegressor

sys.path.append('..')
from utils.downstream import ScoresPredictionTask
from utils.plotting import plot_inter_subject_variability, plot_hard_map
from utils.plotting import plot_global_soft_map

# from utils.estimators import KernelRidgeExt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH_CAMCAN_DATA = Path(
    "/home/mind/alebris/projects/pavi_project/experiments_parcellation/5_camcan_bis/subjects_difumo_128.pkl"
)

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


# %%
def kernel_dice(x, y):
    return (x == y).sum() / len(x)


def kernel_pearson(x, y):
    return pearsonr(x, y).statistic


# %%
@torch.no_grad()
def batch_sample_mean(
    path_run: Union[str, Path],
    subjects_list: List[str],
    epoch_checkpoint: int,
    plate_batch: str,
    n_per_batch: int = 8,
    n_samples: int = 10,
    seed: Optional[int] = 42
) -> Dict:
    # Load the inference module
    
    n_subjects = len(subjects_list)
    path_state = path_run / 'checkpoints' / f'{epoch_checkpoint}_state.pt'
    path_save = path_run / 'checkpoints' / f'{epoch_checkpoint}_mean_sample_post.pt'
    
    # if path_save.exists():
    #     return 0

    inference_module = main(
        subjects_list=subjects_list,
        path_run="",  # No needed here
        seed=seed,
        load_state=path_state,
        return_inference_module=True,
        device=DEVICE
    ).cpu()

    inference_module.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inference_module.guide.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Batch sample the mean
    n_batches = int(np.ceil(inference_module.guide.plate_size[plate_batch] / n_per_batch))
    n_plates = len(inference_module.model.create_plates())
    dim_plate_batch = inference_module.guide.plate_dim[plate_batch]
    data = {}

    print(f'Process job {n_job} with {n_subjects} subjects...')
    for i in tqdm(range(n_batches)):
        idx = {
            plate: torch.arange(inference_module.guide.plate_size[plate])
            for plate in inference_module.guide.plate_size
            if plate != plate_batch
        }

        start = i * n_per_batch
        stop = min((i + 1) * n_per_batch, inference_module.guide.plate_size[plate_batch])
        idx[plate_batch] = torch.arange(start, stop)

        with pyro.plate('vectorize', size=n_samples, dim=- n_plates - 1):
            sample = inference_module(idx=idx)

        if len(data) == 0:
            data = {
                k: v.mean(dim=0)
                for k, v in sample.items()
                if isinstance(v, torch.FloatTensor)    # We ignore the plate samples
            }
        else:
            for k, v_data in data.items():
                if plate_batch in inference_module.guide.sample_plate.get(k, []):
                    data[k] = torch.concat(
                        (v_data, sample[k].mean(dim=0)),
                        dim=n_plates + dim_plate_batch
                    )

    data = {k: v.squeeze() for k, v in data.items()}
    print(f'Process job {n_job} with {n_subjects} subjects - Done')

    torch.save(data, path_save)
    
    return data


# %%
def plot_brain_maps(
    path_run: Union[Path, str],
    epoch_checkpoint: int,
    temperature: float = 1,
):
    path_run = Path(path_run)

    sample = torch.load(
        path_run / 'checkpoints' / f'{epoch_checkpoint}_mean_sample_post.pt',
        map_location='cpu'
    )

    try:
        labels_pop = softmax(sample['theta'].detach().cpu() / temperature, axis=-1)
    except KeyError:
        labels_pop = softmax(sample['theta_s'].detach().cpu().mean(0) / temperature, axis=-1)
    labels = softmax(sample['theta_s'].detach().cpu() / temperature, axis=-1)

    def plot_save(plot_fn, title=None, *args, **kwargs):
        if title is None:
            title = str(plot_fn)
        
        print(f'Start plotting {title}...')
        time1 = time()
        plot_fn(*args, **kwargs)
        (path_run / 'brain_maps').mkdir(exist_ok=True)
        title_file = f'{epoch_checkpoint}_' + title + '.png'
        plt.savefig(str(path_run / 'brain_maps' / title_file))
        plt.close()
        delta_time = time() - time1
        print(f'Done with plotting {title} (elapsed time: {delta_time:.2f}s).')
        
    plots_config = [
        # {
        #     'plot_fn': plot_hard_map,
        #     'title': 'hard_map',
        #     'subjects_soft_map': labels,
        #     'global_soft_map': labels_pop
        # },
        {
            'plot_fn': plot_global_soft_map,
            'title': 'soft_map',
            'global_soft_map': labels_pop,
            'temperature': 0.1
        },
        {
            'plot_fn': plot_inter_subject_variability,
            'title': 'is_variability',
            'subjects_soft_map': labels
        },
    ]
    for config in plots_config:
        plot_save(**config)
      

# %%
def get_downstream_results(
    path_run: Union[Path, str],
    epoch_checkpoint: int,
    subjects_list: List[str],
    seed: Optional[int] = 42
):
    n_subjects = len(subjects_list)

    file_path = os.path.join('/home/mind/jrobador/3_camcan', f'{epoch_checkpoint}_mean_sample_post.pt')
    sample = torch.load(file_path, map_location=DEVICE)

    # Prepare the inputs data
    mu_s = sample['mu_s'].reshape((n_subjects, -1))
    mu_st = sample['mu_st'].mean(1).reshape((n_subjects, -1))
    theta_s = sample['theta_s'].reshape((n_subjects, -1))
    #theta_s_disc = np.argmax(sample['theta_s'], axis=-1)
    #mu_theta = torch.concat((mu_st, theta_s), dim=-1)

    file_path = os.path.join('/home/mind/jrobador/3_camcan', f'{epoch_checkpoint}_state.pt')
    params = torch.load(file_path, map_location=DEVICE)
    
    encod_theta_s = params['guide']['encodings.plate_N_plate_S_unconstrained'].view((n_subjects, -1))
    encod_mu_s = params['guide']['encodings.plate_S_unconstrained'].view((n_subjects, -1))
    encod_mu_st = params['guide']['encodings.plate_S_plate_T_unconstrained'].view((n_subjects, -1))
    encod_mu_theta = torch.concat((encod_mu_st, encod_theta_s), dim=-1)
    
    # Initialize the task and the models
    sp_task = ScoresPredictionTask(
        path_run=str(path_run),
        path_scores=Path('./scores_camcan.csv'),
        subjects=subjects_list,
        categories=categories,
        fancy_categories_names=fancy_categories
    )

    inputs_models = {
        #'mu_s': {
        #    'data': mu_s,
        #    'models': {
        #        # 'pearson_ridge_kernel': KernelRidge(kernel=kernel_pearson),
        #        #'laplacian_ridge_kernel': KernelRidge(kernel='laplacian'),
        #        # 'cosine_ridge_kernel': KernelRidge(kernel='cosine'),
        #        'ridge_regression': Ridge(),
        #        'custom_ridge': None,
        #    }
        #},
        #'mu_st': {
        #    'data': mu_st,
        #    'models': {
        #        # 'pearson_ridge_kernel': KernelRidge(kernel=kernel_pearson),
        #        #'laplacian_ridge_kernel': KernelRidge(kernel='laplacian'),
        #        # 'cosine_ridge_kernel': KernelRidge(kernel='cosine'),
        #        'ridge_regression': Ridge(),
        #        'custom_ridge': None,
        #    }
        #},
        'theta_s': {
            'data': theta_s,
            'models': {
               # 'ridge_regression': Ridge(),
               'custom_ridge': None,
                #'svr':None,
            }
        },
        #'theta_s_disc': {
        #    'data': theta_s_disc,
        #    'models': {
        #        'dice_ridge_kernel': KernelRidge(kernel=kernel_dice),
        #        'custom_ridge': None,
        #    }
        #},
        #'encod_theta_s': {
        #    'data': encod_theta_s,
        #    'models': {
        #        # 'laplacian_ridge_kernel': KernelRidge(kernel='laplacian'),
        #        # 'cosine_ridge_kernel': KernelRidge(kernel='cosine'),
        #        'ridge_regression': Ridge(),
        #        'custom_ridge': None,
        #    }
        #},
        #'encod_mu_s': {
        #    'data': encod_mu_s,
        #    'models': {
        #        'laplacian_ridge_kernel': KernelRidge(kernel='laplacian'),
        #        # 'cosine_ridge_kernel': KernelRidge(kernel='cosine'),
        #        'ridge_regression': Ridge(),
        #        'custom_ridge': None,
        #    }
        #},
        # 'encod_mu_st': {
        #     'data': encod_mu_st,
        #     'models': {
        #         'laplacian_ridge_kernel': KernelRidge(kernel='laplacian'),
        #         # 'cosine_ridge_kernel': KernelRidge(kernel='cosine'),
        #         'ridge_regression': Ridge(),
        #     }
        # },
        # 'mu_theta': {
        #     'data': mu_theta,
        #     'models': {
        #         'ridge_regression': Ridge(),
        #         'custom_ridge': None,
        #     }
        # },
        # 'encod_mu_theta': {
        #     'data': encod_mu_theta,
        #     'models': {
        #         'ridge_regression': Ridge(),
        #     }
        # },
    }

    # Fit and predict the results
    for name_input in inputs_models:
        X = inputs_models[name_input]['data']
        models = inputs_models[name_input]['models']
        
        sp_task.predict(
            X=X,
            models=models,
            name_X=name_input + f'_{epoch_checkpoint}',
            n_pca_components=190,
            n_splits=10,
            n_repeats=5,
            n_jobs=1,
            random_state=seed,
            plot=True,
        )

    # Save the results
    directory_path_golden = Path("/home/mind/jrobador/pavi_postprocess/golden_model")
    directory_path_custom = Path("/home/mind/jrobador/pavi_postprocess/custom_entire_model")
    sp_task.results.to_csv(directory_path_custom / f'downstream_results_{epoch_checkpoint}_ext.csv')


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_job', type=int, required=True, help='ID of the job')
    args = parser.parse_args()
    n_job = args.n_job
    
    # n_job = 40
    
    with open('jobs.json', 'r') as f:
        jobs = json.load(f)
    seed = jobs['seed'][n_job]
    subjects_list = jobs['sub_list'][n_job]
    
    # with open(PATH_CAMCAN_DATA, 'rb') as f:
    #     xr_data = pickle.load(f)
    # subjects_list = xr_data.Subject.values
    
    epoch_checkpoint_list = [12]
    # path_run = Path(f'./runs_warmbayes/transfer_learning_{n_job}_S{len(subjects_list)}/')
    # path_run = Path(f'./runs/inference_{n_job}_S{len(subjects_list)}/')
    # path_run = Path('./runs/inference_S553_XR/')
    list_path_run = [
        # Path(f'./runs_warmbayes/transfer_learning_{n_job}_S{len(subjects_list)}/'),
        Path(f'/home/mind/alebris/projects/pavi_project/experiments_parcellation/3_camcan/runs_warmbayes/transfer_learning_{n_job}_S{len(subjects_list)}/')
    ]
    print (list_path_run)
    for path_run in list_path_run:
        for epoch_checkpoint in epoch_checkpoint_list:
            print("-------------------------------")
            print(f'Processing epoch {epoch_checkpoint}')
            # batch_sample_mean(
            #     subjects_list=subjects_list,
            #     path_run=path_run,
            #     epoch_checkpoint=epoch_checkpoint,
            #     plate_batch='plate_S',
            #     n_per_batch=10,
            #     n_samples=10,
            # )
            # plot_brain_maps(
            #     epoch_checkpoint=epoch_checkpoint,
            #     path_run=path_run
            # )
            get_downstream_results(
                epoch_checkpoint=epoch_checkpoint,
                path_run=path_run,
                subjects_list=subjects_list
            )
            print("-------------------------------")

# %%
