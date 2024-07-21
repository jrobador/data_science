import skorch
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class MultiModalEmbedding(nn.Module):
    def __init__(self, f_input_dim, c_input_dim, f_layers_dim, c_layers_dim, latent_space_dim, alpha=1):
        super(MultiModalEmbedding, self).__init__()
        self.alpha = alpha

        # Features Encoder
        self.f_encoder = nn.Sequential(
            nn.Linear(f_input_dim, f_layers_dim[0]),
            nn.ReLU(),
            # nn.BatchNorm1d(f_layers_dim[0]),
            # nn.Dropout(0.2)
        )
        for i in range(len(f_layers_dim) - 1):
            self.f_encoder.add_module(f'linear_{i}', nn.Linear(f_layers_dim[i], f_layers_dim[i+1]))
            self.f_encoder.add_module(f'leaky_relu_{i}', nn.ReLU())
            # self.f_encoder.add_module(f'batch_norm_{i}', nn.BatchNorm1d(f_layers_dim[i+1]))
            # self.f_encoder.add_module(f'dropout_{i}', nn.Dropout(0.2))
        self.f_encoder.add_module(f'latent_space', nn.Linear(f_layers_dim[-1], latent_space_dim))

        # Cognition Encoder
        self.c_encoder = nn.Sequential(
            nn.Linear(c_input_dim, c_layers_dim[0]),
            nn.ReLU(),
            #nn.BatchNorm1d(c_layers_dim[0]),
            # nn.Dropout(0.2)
        )
        for i in range(len(c_layers_dim) - 1):
            self.c_encoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i+1]))
            self.c_encoder.add_module(f'leaky_relu_{i}', nn.ReLU())
            #self.c_encoder.add_module(f'batch_norm_{i}', nn.BatchNorm1d(c_layers_dim[i+1]))
            # self.c_encoder.add_module(f'dropout_{i}', nn.Dropout(0.2))
        self.c_encoder.add_module(f'latent_space', nn.Linear(c_layers_dim[-1], latent_space_dim))

        # Cognition Decoder
        self.c_decoder = nn.Sequential(
            nn.Linear(latent_space_dim, c_layers_dim[-1]),
            nn.ReLU(),
            # nn.BatchNorm1d(c_layers_dim[-1]),
            # nn.Dropout(0.2)
        )
        for i in range(len(c_layers_dim) - 1, 0, -1):
            self.c_decoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i-1]))
            self.c_decoder.add_module(f'leaky_relu_{i}', nn.ReLU())
            # self.c_decoder.add_module(f'batch_norm_{i}', nn.BatchNorm1d(c_layers_dim[i-1]))
            # self.c_decoder.add_module(f'dropout_{i}', nn.Dropout(0.2))
        self.c_decoder.add_module(f'input_space', nn.Linear(c_layers_dim[0], c_input_dim))


    def forward (self, x, y):
        f_gamma = self.f_encoder(x)
        g_beta  = self.c_encoder(y)
        h_eta   = self.c_decoder(g_beta)

        return f_gamma, g_beta, h_eta
    

    def predict_y_from_x(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(next(self.parameters()).device)
        with torch.no_grad():
            f_gamma = self.f_encoder(x)
            h_eta_f_gamma = self.c_decoder(f_gamma)
        
        return h_eta_f_gamma
    


class MultiModalEmbeddingNet(skorch.NeuralNet):    
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        f_gamma, g_beta, h_eta = y_pred
        l2_norm = super().get_loss(f_gamma, g_beta, *args, **kwargs)
        reconst_loss = self.module_.alpha * super().get_loss(h_eta, y_true, *args, **kwargs)
        loss = l2_norm + reconst_loss
        self.history.record('l2_norm', l2_norm.item())
        self.history.record('reconst_loss', reconst_loss.item())

        return loss
    

    def score(self, x, y):
        x_values = []
        y_values = []

        for sample in x:
            data_dict, y = sample
            x_values.append(data_dict['x'])
            y_values.append(y)

        x_values = np.array(x_values)
        y_values = np.array(y_values)
        
        y_pred = self.module_.predict_y_from_x(x_values).cpu().numpy()
        mape = mean_absolute_percentage_error(y_values,y_pred)
        self.history.record('mape', mape.item())

        return mape
    


class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    

class RidgeRegressionWDimRed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=1):
        super(RidgeRegressionWDimRed, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        self.alpha = alpha

    def orthogonal_regularization(self):
        rank_transform = self.net[0].weight @ self.net[0].weight.T
        identity = torch.eye(rank_transform.shape[0]).to(rank_transform.device)
        return ((rank_transform - identity) ** 2).sum()

    def l2_regularization(self):
        l2_norm = torch.norm(self.net[1].weight)
        return self.alpha * l2_norm
    
    def forward(self, x):
        return self.net(x) 

class RidgeNet(skorch.NeuralNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        loss += self.module_.l2_regularization() + self.module_.orthogonal_regularization()
        return loss
    

                

# custom_model = NeuralNetRegressor(
#                 module=RidgeRegression,
#                 module__input_dim=input_dim,
#                 module__output_dim=output_dim,
#                 optimizer__weight_decay=alpha,
#                 criterion=nn.MSELoss,
#                 optimizer=torch.optim.AdamW,
#                 optimizer__lr=0.001,
#                 max_epochs=7500,
#                 device='cuda' if torch.cuda.is_available() else 'cpu',
#             )

#hidden_dim = 1500
# custom_model = RidgeNet(
#                 module=RidgeRegressionWDimRed,
#                 module__input_dim=input_dim,
#                 module__hidden_dim=hidden_dim,
#                 module__output_dim=output_dim,
#                 module__alpha=alpha,
#                 criterion=nn.MSELoss,
#                 optimizer=torch.optim.AdamW,
#                 optimizer__lr=0.0001,
#                 callbacks=[(    'lr_scheduler', 
#                                 LRScheduler(policy=ReduceLROnPlateau,
#                                                        mode='min', 
#                                                        factor=0.1, 
#                                                        patience=10)
#                             ),
#                             (   'early_stopping',
#                                 EarlyStopping(monitor='valid_loss', 
#                                               patience=12, 
#                                               threshold=0.0002)
#                                 
#                             ),
#                             #(   'best_model_saving',
#                             #    Checkpoint(
#                             #        monitor='valid_loss_best',
#                             #        f_pickle="best_model_{last_epoch[epoch]}.pt",
#                             #    ),
#                             #)
#                         ],
#                 max_epochs=1000,
#                 device='cuda:0' if torch.cuda.is_available() else 'cpu',
#             )