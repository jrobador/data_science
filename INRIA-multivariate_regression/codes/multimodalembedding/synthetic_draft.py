import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from aux import plot_latent


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


# Definir la clase MultiModalEmbedding
class MultiModalEmbedding(nn.Module):
    def __init__(self, f_input_dim, c_input_dim, f_layers_dim, c_layers_dim, latent_space_dim, alpha=1):
        super(MultiModalEmbedding, self).__init__()
        self.alpha = alpha

        # Features Encoder
        self.f_encoder = nn.Sequential(
            nn.Linear(f_input_dim, f_layers_dim[0]),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        for i in range(len(f_layers_dim) - 1):
            self.f_encoder.add_module(f'linear_{i}', nn.Linear(f_layers_dim[i], f_layers_dim[i+1]))
            self.f_encoder.add_module(f'relu_{i}', nn.ReLU())
            self.f_encoder.add_module(f'dropout_{i}', nn.Dropout(0.15))
        self.f_encoder.add_module('latent_space', nn.Linear(f_layers_dim[-1], latent_space_dim))

        # Cognition Encoder
        self.c_encoder = nn.Sequential(
            nn.Linear(c_input_dim, c_layers_dim[0]),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        for i in range(len(c_layers_dim) - 1):
            self.c_encoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i+1]))
            self.c_encoder.add_module(f'relu_{i}', nn.ReLU())
            self.c_encoder.add_module(f'dropout_{i}', nn.Dropout(0.15))
        self.c_encoder.add_module('latent_space', nn.Linear(c_layers_dim[-1], latent_space_dim))

        # Cognition Decoder
        self.c_decoder = nn.Sequential(
            nn.Linear(latent_space_dim, c_layers_dim[-1]),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        for i in range(len(c_layers_dim) - 1, 0, -1):
            self.c_decoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i-1]))
            self.c_decoder.add_module(f'relu_{i}', nn.ReLU())
            self.c_decoder.add_module(f'dropout_{i}', nn.Dropout(0.15))
        self.c_decoder.add_module('input_space', nn.Linear(c_layers_dim[0], c_input_dim))

    def forward(self, x, y):
        f_gamma = self.f_encoder(x)
        g_beta = self.c_encoder(y)
        h_eta = self.c_decoder(g_beta)
        return f_gamma, g_beta, h_eta

    def predict_y_from_x(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(next(self.parameters()).device)
        with torch.no_grad():
            f_gamma = self.f_encoder(x)
            h_eta_f_gamma = self.c_decoder(f_gamma)
        return h_eta_f_gamma

# Definir la función de pérdida
def custom_loss(y_pred, y_true, model):
    f_gamma, g_beta, h_eta = y_pred
    l2_norm = nn.MSELoss()(f_gamma, g_beta)
    reconst_loss = model.alpha * nn.MSELoss()(h_eta, y_true)
    loss = l2_norm + reconst_loss
    return loss

# Entrenamiento del modelo
def train_model(model, train_loader, num_epochs=100, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch, y_batch)
            loss = custom_loss(y_pred, y_batch, model)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Datos de ejemplo (asegúrate de usar tus propios datos)
f_input_dim = 190
c_input_dim = 13
f_layers_dim = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 32]
c_layers_dim = [11, 9]
latent_space_dim = 6
alpha = 1

num_samples = 1000
x_data = np.random.randn(num_samples, f_input_dim)
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

# Convertir a tensores de PyTorch
x_train = torch.tensor(x_data, dtype=torch.float32)
y_train = torch.tensor(y_data, dtype=torch.float32)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar y entrenar el modelo
model = MultiModalEmbedding(f_input_dim, c_input_dim, f_layers_dim, c_layers_dim, latent_space_dim, alpha)
model.to(device)
train_model(model, train_loader)

def generate_predictions(model, data_loader):
    model.eval()
    f_gamma_pred_list = []
    g_beta_pred_list = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            f_gamma_pred, g_beta_pred, _ = model(x_batch, y_batch)
            f_gamma_pred_list.append(f_gamma_pred.cpu())
            g_beta_pred_list.append(g_beta_pred.cpu())
    f_gamma_pred = torch.cat(f_gamma_pred_list)
    g_beta_pred = torch.cat(g_beta_pred_list)
    return f_gamma_pred, g_beta_pred

f_gamma_pred, g_beta_pred = generate_predictions(model,train_loader)

print (f"{f_gamma_pred=}, \n {g_beta_pred=}")

plot_latent(f_gamma_pred, g_beta_pred, './', 'syntethic')
