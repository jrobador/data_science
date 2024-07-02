import torch.nn as nn
import skorch
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import pickle
import torch.utils as utils
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats.mstats import pearsonr
import matplotlib.pyplot as plt


class MultiModalEmbedding(nn.Module):
    def __init__(self, f_input_dim, c_input_dim, f_layers_dim, c_layers_dim, latent_space_dim, alpha=1):
        super(MultiModalEmbedding, self).__init__()
        self.alpha = alpha

        #Features Encoder
        self.f_encoder = nn.Sequential(
            nn.Linear(f_input_dim, f_layers_dim[0]),
            nn.ReLU()
        )
        for i in range(len(f_layers_dim) - 1):
            self.f_encoder.add_module(f'linear_{i}', nn.Linear(f_layers_dim[i], f_layers_dim[i+1]))
            self.f_encoder.add_module(f'relu_{i}', nn.ReLU())
        self.f_encoder.add_module(f'latent_space', nn.Linear(f_layers_dim[-1], latent_space_dim))

        #Cognition Encoder
        self.c_encoder = nn.Sequential(
            nn.Linear(c_input_dim, c_layers_dim[0]),
            nn.ReLU()
        )
        for i in range(len(c_layers_dim) - 1):
            self.c_encoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i+1]))
            self.c_encoder.add_module(f'relu_{i}', nn.ReLU())
        self.c_encoder.add_module(f'latent_space', nn.Linear(c_layers_dim[-1], latent_space_dim))

        #Cognition Decoder
        self.c_decoder = nn.Sequential(
            nn.Linear(latent_space_dim, c_layers_dim[-1]),
            nn.ReLU(),
        )
        for i in range (len(c_layers_dim) - 1, 0, -1):
            self.c_decoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i-1]))
            self.c_decoder.add_module(f'relu_{i}', nn.ReLU())
        self.c_decoder.add_module(f'input_space', nn.Linear(c_layers_dim[0], c_input_dim))

    def forward (self, x, y):
        f_gamma = self.f_encoder(x)
        g_beta  = self.c_encoder(y)
        h_eta   = self.c_decoder(g_beta)

        return f_gamma, g_beta, h_eta


class MultiModalEmbeddingNet(skorch.NeuralNetRegressor):    
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        f_gamma, g_beta, h_eta = y_pred
        
        l2_norm = super().get_loss(f_gamma, g_beta, *args, **kwargs)
        reconst_loss = super().get_loss(h_eta, y_true, *args, **kwargs)
        loss = l2_norm + self.module_.alpha * reconst_loss 

        return loss
    

def extract_data(cognitive_variables):
    data_real = pickle.load(
        open(
            '/data/parietal/store2/work/ggomezji/graph_dmri/data/subjects_1500_LH_non_neighbours_linear_interpolation_concatenated.pkl',
            'rb'
        )
    )
    attenuations, cognition = list(zip(*[
        (d.x.squeeze(), torch.Tensor([d[c] for c in cognitive_variables]))
        for d in data_real
    ]))
    attenuations = torch.nan_to_num(torch.stack(attenuations).to(torch.float32))
    cognition = torch.nan_to_num(torch.stack(cognition).to(torch.float32))

    return attenuations, cognition


def real_data(cognitive_variables):
    attenuations, cognition = extract_data(cognitive_variables)
    subject_data = utils.data.TensorDataset(attenuations, cognition)
    train_set, validation_set = utils.data.random_split(
        subject_data, [.8, .2]
    )

    X_train = torch.stack([item[0] for item in train_set])
    Y_train = torch.stack([item[1] for item in train_set])
    X_test = torch.stack([item[0] for item in validation_set])
    Y_test = torch.stack([item[1] for item in validation_set])

    return X_train, Y_train, X_test, Y_test


# def compute_pearson_scores(model, categories, X_test, Y_test):
#     Y_pred = model.predict(X_test)
#     n_scores = len(categories)
#     scores_ = np.full((n_scores,), np.nan)
#     
#     for c in range(n_scores):
#         r, _ = pearsonr(Y_pred[:, c], Y_test[:, c])
#         scores_[c] = r
#         
#     return scores_


# def plot_scores_prediction(y, title='', name_file=''):
#         mean = y.to_numpy().mean()
#         std = y.to_numpy().mean(axis=1).std()
# 
#         fig, ax = plt.subplots()
#         sns.boxplot(
#             data=y, width=0.3, color='skyblue', ax=ax, showmeans=True,
#             meanprops={
#                 "marker": "o",
#                 "markerfacecolor": "white",
#                 "markeredgecolor": "black",
#                 "markersize": "5"
#             }
#         )
#         plt.xticks(rotation=45, ha='right')
#         plt.axhline(y=0, color='grey', linestyle='dashed')
#         # plt.ylim(-0.5, 0.5)
#         plt.ylabel('r-Pearson correlation')
#         plt.title(title + f"\n Mean score: {mean:.4f} +/- {std:.4f}")
#         plt.tight_layout()
# 
#         plt.savefig(name_file + '.png')   


def main():
    
    cognitive_variables = [
        'Age_in_Yrs', 
        'WM_Task_Acc', 
        'WM_Task_Median_RT',
        'Relational_Task_Acc',
        'Relational_Task_Median_RT', 
        'Gambling_Task_Perc_Larger',
        'Gambling_Task_Median_RT_Larger',
        'ListSort_AgeAdj', 
        'Flanker_AgeAdj',
        'CardSort_AgeAdj',
        'PicSeq_AgeAdj',
        'ProcSpeed_AgeAdj'
    ]

    #Dataset loader
    X_train, Y_train, X_test, Y_test = real_data(cognitive_variables)
    
    fwd_dict = {'x': X_train, 'y': Y_train}

    #Model parameters

    f_input_dim = X_train.shape[1]
    c_input_dim = Y_train.shape[1]

    print (f_input_dim, c_input_dim)

    f_layers_dim = [8192, 1024, 128]
    c_layers_dim = [10, 8]
    latent_space_dim = 6

    alpha = 0.9

    model = MultiModalEmbeddingNet (

        module = MultiModalEmbedding,
        module__f_input_dim = f_input_dim,
        module__c_input_dim = c_input_dim,
        module__f_layers_dim = f_layers_dim,
        module__c_layers_dim = c_layers_dim,
        module__latent_space_dim = latent_space_dim,
        module__alpha = alpha,

        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=0.01,
        max_epochs=25,

        callbacks = [('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,
                                                  mode='min', 
                                                  factor=0.1, 
                                                  patience=10
                                                 )),
                     ('early_stopper', EarlyStopping(monitor='valid_loss',
                                                    patience=100,
                                                    threshold=0.0001
                                                    )),
                     #('best_model_saving', Checkpoint(
                     #                           monitor='valid_loss_best',
                     #                           f_pickle="best_model_{last_epoch[epoch]}.pt",
                     #                       )),                             
                    ],

        device='cuda:0' if torch.cuda.is_available() else 'cpu',
    )


    model.fit(fwd_dict, Y_train)

    predicted_scores = compute_pearson_scores(model, cognitive_variables, X_test, Y_test)


    df_pred = pd.DataFrame(predicted_scores, columns=cognitive_variables)
    df_pred = df_pred.reset_index()
    df_pred['index'] = df_pred['index'] // 1
    df_pred = df_pred.groupby('index').mean()


    plot_scores_prediction(df_pred, title='Pearson Correlation Scores', name_file='pearson_scores')

if __name__ == '__main__':
    main()



