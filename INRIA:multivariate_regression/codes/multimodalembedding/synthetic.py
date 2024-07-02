import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from networks import MultiModalEmbedding, MultiModalEmbeddingNet
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, TensorBoard, EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from aux import validation, plot_latent

# Función para fijar las semillas
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fijar semilla
seed = 42
set_seed(seed)

# Definimos dimensiones de entrada y capas
f_input_dim = 190
c_input_dim = 13
f_layers_dim = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]
c_layers_dim = [12,11,10,9,8,7,6]
latent_space_dim = 4
num_samples = 10000
x_data = np.random.randn(num_samples, f_input_dim).astype(np.float32)

# Definir una relación no lineal para generar las etiquetas (y)
# Aquí usamos una combinación de funciones trigonométricas y polinomiales
def generate_y(x):
    y = np.zeros((x.shape[0], c_input_dim))
    for i in range(c_input_dim):
        y[:, i] = (
            np.sin(x[:, i % f_input_dim]) +
            np.cos(x[:, (i + 1) % f_input_dim]) +
            np.tan(x[:, (i + 2) % f_input_dim]) +
            np.power(x[:, (i + 3) % f_input_dim], 2) +
            0.1 * np.random.randn(x.shape[0])  # Añadir algo de ruido
        )
    return y

y_data = generate_y(x_data)

# Normalización de los datos
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_data = scaler_x.fit_transform(x_data)
y_data = scaler_y.fit_transform(y_data)


print(f"{x_data=}, {x_data.shape} \n {y_data=}, {y_data.shape}")

# Convertimos a tensores y los transformamos a float32
x_tensor = torch.tensor(x_data).float()
y_tensor = torch.tensor(y_data).float()

fwd_dict = {'x': x_tensor, 
            'y': y_tensor}

model = MultiModalEmbeddingNet (
    module = MultiModalEmbedding,
    module__f_input_dim = f_input_dim,
    module__c_input_dim = c_input_dim,
    module__f_layers_dim = f_layers_dim,
    module__c_layers_dim = c_layers_dim,
    module__latent_space_dim = latent_space_dim,
    module__alpha = 1,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=0.001,
    max_epochs=2500,
    #train_split=None,
    batch_size=32,
    callbacks = [('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,
                                              mode='min', 
                                              factor=0.1, 
                                              patience=75
                                             )),
                 ('early_stopper', EarlyStopping(monitor='train_loss',
                                                patience=100,
                                                threshold=0.0001
                                                )),
    #              ('best_model_saving', Checkpoint(
    #                                         monitor='valid_loss_best',
    #                                         f_pickle="best_model_{last_epoch[epoch]}.pt",
    #                                     )),     
                    ('mape_scoring', EpochScoring(scoring=None, lower_is_better=True, on_train=True))
                 ],
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
)


model.fit(fwd_dict, y_tensor)

f_gamma_pred, g_beta_pred, h_eta_pred  = validation(model, fwd_dict)
plot_latent(f_gamma_pred,g_beta_pred, './', 'synthetic')
