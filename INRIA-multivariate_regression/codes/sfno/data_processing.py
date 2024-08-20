import torch
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import hcp_utils as hcp


def dataset_preprocessing(DEVICE, file_path, path_jobs, path_scores, n_job=44):

    sample = torch.load(file_path, map_location=DEVICE)
    temperature = 1

    ## Para la regresion no se necesita el softmax!
    #data = softmax(sample['theta_s'].detach().cpu() / temperature, axis=-1)
    data = sample['theta_s'].detach().cpu()

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

    X = np.array([x.cpu().numpy() for x in data])[scores_mask]
    y = scores.to_numpy()
    
    return X, y, seed 

def data_with_mask(X, net_number):
    vertices = hcp.mesh['sphere'][0] / np.linalg.norm(hcp.mesh['sphere'][0], axis=1, keepdims=True)
    vertices_left  = vertices[:32492]
    vertices_right = vertices[32492:]

    mask = hcp.cortex_data(np.ones(len(vertices))).astype(bool)
    data_cortex = np.zeros((X.shape[0], len(mask), X.shape[-1]))

    data_cortex[:, mask, :] = X

    if net_number is None:
        network_left = data_cortex[:, :len(vertices_left), :]  
        network_right = data_cortex[:, len(vertices_right):, :]
    elif isinstance(net_number, int):
        network_left = data_cortex[:, :len(vertices_left), net_number]  
        network_right = data_cortex[:, len(vertices_right):, net_number]
    else:
        raise ValueError("net_number must be an integer or None")

    return vertices_left, vertices_right, network_left, network_right


