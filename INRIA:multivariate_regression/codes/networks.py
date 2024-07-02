import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class MultiModalEmbedding(nn.Module):
    def __init__(self, f_input_dim, c_input_dim, f_layers_dim, c_layers_dim, latent_space_dim, alpha=1):
        super(MultiModalEmbedding, self).__init__()
        self.alpha = alpha

        # Features Encoder
        self.f_encoder = nn.Sequential(
            nn.Linear(f_input_dim, f_layers_dim[0]),
            nn.GELU(),
        )
        for i in range(len(f_layers_dim) - 1):
            self.f_encoder.add_module(f'linear_{i}', nn.Linear(f_layers_dim[i], f_layers_dim[i+1]))
            self.f_encoder.add_module(f'relu_{i}', nn.GELU())
        self.f_encoder.add_module(f'latent_space', nn.Linear(f_layers_dim[-1], latent_space_dim))

        # Cognition Encoder
        self.c_encoder = nn.Sequential(
            nn.Linear(c_input_dim, c_layers_dim[0]),
            nn.GELU(),
        )
        for i in range(len(c_layers_dim) - 1):
            self.c_encoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i+1]))
            self.c_encoder.add_module(f'relu_{i}', nn.GELU())
        self.c_encoder.add_module(f'latent_space', nn.Linear(c_layers_dim[-1], latent_space_dim))

        # Cognition Decoder
        self.c_decoder = nn.Sequential(
            nn.Linear(latent_space_dim, c_layers_dim[-1]),
            nn.GELU(),
        )
        for i in range(len(c_layers_dim) - 1, 0, -1):
            self.c_decoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i-1]))
            self.c_decoder.add_module(f'relu_{i}', nn.GELU())
        self.c_decoder.add_module(f'input_space', nn.Linear(c_layers_dim[0], c_input_dim))


    def forward (self, x, y):
        f_gamma = self.f_encoder(x)
        g_beta  = self.c_encoder(y)
        h_eta   = self.c_decoder(g_beta)
        h_eta_f_gamma = self.c_decoder(f_gamma)

        return f_gamma, g_beta, h_eta, h_eta_f_gamma
    

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
        f_gamma, g_beta, h_eta, h_eta_f_gamma = y_pred
        #l2_norm = super().get_loss(f_gamma, g_beta, *args, **kwargs)
        # self.history.record('l2_norm', l2_norm.item())
        
        reconst_loss = super().get_loss(h_eta, h_eta_f_gamma, *args, **kwargs)
        self.history.record('reconst_loss', reconst_loss.item())
        y_true_loss  = super().get_loss(h_eta, y_true, *args, **kwargs)
        self.history.record('y_true_loss', y_true_loss.item())
        y_from_x_loss = super().get_loss(h_eta_f_gamma, y_true, *args, **kwargs)
        self.history.record('y_from_x_loss', y_from_x_loss.item())

               

        loss = reconst_loss+ y_true_loss  + y_from_x_loss

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
class RidgeRegressionNet(skorch.NeuralNetRegressor):
    def score(self, x, y):
        y_pred = self.predict(x)
        mape = - mean_absolute_percentage_error(y,y_pred)
        self.history.record('mape_test', mape.item())

        return mape



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
class RidgeRegressionWDimRedNet(skorch.NeuralNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        loss += self.module_.l2_regularization() + self.module_.orthogonal_regularization()
        return loss
    



class NonlinearRidgeRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(NonlinearRidgeRegression, self).__init__()
        
        self.mlp = nn.Sequential(
            (nn.Linear(input_dim, hidden_dims[0])),
            (nn.ReLU()),
        )
        for i in range(1, len(hidden_dims)):
            self.mlp.add_module(f'linear_{i}', nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
        self.mlp.add_module(f'output_dim', nn.Linear(hidden_dims[-1], output_dim))
        
    def forward(self, x):
        x = self.mlp(x)
        return x
    
class NonlinearRidgeRegressionNet(skorch.NeuralNetRegressor):
    def score(self, x, y):
        y_pred = self.predict(x)
        mape = - mean_absolute_percentage_error(y, y_pred)
        self.history.record('mape_test', mape.item())
        return mape







class IdentityBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(IdentityBlock, self).__init__()
        self.units = units
        self.block = nn.Sequential(
            nn.Linear(input_dim, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity 
        out = F.relu(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseBlock, self).__init__()
        self.units = units
        self.block = nn.Sequential(
            nn.Linear(input_dim, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units)
        )
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, units),
            nn.BatchNorm1d(units)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        shortcut = self.shortcut(identity)
        out += shortcut
        out = F.relu(out)
        return out

class ResNet50Regression(nn.Module):
    def __init__(self, input_dim, output_dim, width=64):
        super(ResNet50Regression, self).__init__()
        self.width = width
        # Sequential blocks
        self.dense_blocks = nn.Sequential(
            DenseBlock(input_dim, width),
            IdentityBlock(width, width),
            IdentityBlock(width, width),

            DenseBlock(width, width),
            IdentityBlock(width, width),
            IdentityBlock(width, width),

            DenseBlock(width, width),
            IdentityBlock(width, width),
            IdentityBlock(width, width)
        )
        # Final layers
        self.final_layers = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, output_dim)
        )
    def forward(self, x):
        x = self.dense_blocks(x)
        x = self.final_layers(x)
        return x


class ResNet50RegressionNet(skorch.NeuralNetRegressor):
    def score(self, x, y):
        y_pred = self.predict(x)
        mape = - mean_absolute_percentage_error(y, y_pred)
        self.history.record('mape', mape.item())
        return mape










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



##############
#                 f_input_dim = input_dim
#                 c_input_dim = output_dim
#                 f_layers_dim = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
#                 c_layers_dim = [ 12, 11, 10, 9, 8, 7]
#                 latent_space_dim = 6
#                custom_model = MultiModalEmbeddingNet (
#                    module = MultiModalEmbedding,
#                    module__f_input_dim = f_input_dim,
#                    module__c_input_dim = c_input_dim,
#                    module__f_layers_dim = f_layers_dim,
#                    module__c_layers_dim = c_layers_dim,
#                    module__latent_space_dim = latent_space_dim,
#                    module__alpha = alpha,
#
#                    criterion=nn.MSELoss,
#                    optimizer=torch.optim.AdamW,
#                    optimizer__lr=0.0001,
#                    max_epochs=1000,
#                    #train_split=None,
#                    batch_size=-1,
#
#                    callbacks = [   ('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,
#                                                              mode='min', 
#                                                              factor=0.1, 
#                                                              patience=50
#                                                             )),
#                                    ('early_stopper', EarlyStopping(monitor='train_loss',
#                                                                 patience=100,
#                                                                 threshold=0.0001
#                                                                 )),
#                    #              #('best_model_saving', Checkpoint(
#                    #              #                           monitor='valid_loss_best',
#                    #              #                           f_pickle="best_model_{last_epoch[epoch]}.pt",
#                    #              #                       )),
#                                    ('tensorboard', TensorBoard(writer)),
#                                    ('mape_scoring', EpochScoring(scoring=None, lower_is_better=True, on_train=True))                             
#                                 ],
#
#                    device='cuda:0' if torch.cuda.is_available() else 'cpu',
#                )
                # Plotting latent space
#                f_gamma_train, g_beta_train, _ = validation(custom_model, fwd_dict)
#                plot_latent(f_gamma_pred=f_gamma_train, g_beta_pred=g_beta_train, dir="/home/mind/jrobador/pavi_multimodal_results", name=("train_" + name))
#                f_gamma_test, g_beta_test, _  = validation(custom_model, test_dict)
#                plot_latent(f_gamma_pred=f_gamma_test, g_beta_pred=g_beta_test, dir="/home/mind/jrobador/pavi_multimodal_results", name=("test_" + name))


#################



                #hidden_dim=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
                #custom_model = NonlinearRidgeRegressionNet(
                #                module=NonlinearRidgeRegression,
                #                module__input_dim=input_dim,
                #                module__hidden_dims=hidden_dim,
                #                module__output_dim=output_dim,
                #                optimizer__weight_decay=alpha,
                #                criterion=nn.MSELoss,
                #                optimizer=torch.optim.AdamW,
                #                optimizer__lr=0.01,
                #                max_epochs=7500,
                #                device='cuda' if torch.cuda.is_available() else 'cpu',
                #                callbacks = [('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,mode='min', factor=0.1, patience=50)),
                #                             ('mape_scoring_train', EpochScoring(scoring='neg_mean_absolute_percentage_error', lower_is_better=True, on_train=True)),
                #                             ('mape_scoring_test',  EpochScoring(scoring=None, lower_is_better=True, on_train=False)),
                #                             ('early_stopper', EarlyStopping(monitor='train_loss',
                #                                                patience=1000,
                #                                                threshold=0.0001
                #                                                )),
                #                             ]
                #            )
#
                #print(f"{hidden_dim=}")