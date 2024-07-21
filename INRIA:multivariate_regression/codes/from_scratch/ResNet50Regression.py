import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import skorch
from sklearn.metrics import mean_absolute_percentage_error
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import LearningCurveDisplay


from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, TensorBoard, EpochScoring

import os
import sys
sys.path.append('/home/mind/jrobador/multimodalembedding')

from metrics import r2_test_score, r2_train_score, mape_train_score, mape_test_score, mae_train_score, mae_test_score

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


class IdentityBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(IdentityBlock, self).__init__()
        self.units = units
        self.fc1 = nn.Linear(input_dim, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(units, units)
        self.bn2 = nn.BatchNorm1d(units)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(units, units)
        self.bn3 = nn.BatchNorm1d(units)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.bn3(out)

        out += identity  # Residual connection
        out = F.relu(out)

        return out

class DenseBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseBlock, self).__init__()
        self.units = units
        self.fc1 = nn.Linear(input_dim, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(units, units)
        self.bn2 = nn.BatchNorm1d(units)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(units, units)
        self.bn3 = nn.BatchNorm1d(units)

        self.shortcut_fc = nn.Linear(input_dim, units)
        self.shortcut_bn = nn.BatchNorm1d(units)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.bn3(out)

        identity = self.shortcut_fc(identity)
        identity = self.shortcut_bn(identity)

        out += identity  # Residual connection
        out = F.relu(out)

        return out

class ResNet50Regression(nn.Module):
    def __init__(self, input_dim, output_dim, width=512):
        super(ResNet50Regression, self).__init__()
        self.width = width
        self.dense_block1 = DenseBlock(input_dim, width)
        self.identity_block1 = IdentityBlock(width, width)
        self.identity_block2 = IdentityBlock(width, width)

        self.dense_block2 = DenseBlock(width, width)
        self.identity_block3 = IdentityBlock(width, width)
        self.identity_block4 = IdentityBlock(width, width)

        self.dense_block3 = DenseBlock(width, width)
        self.identity_block5 = IdentityBlock(width, width)
        self.identity_block6 = IdentityBlock(width, width)

        self.final_bn = nn.BatchNorm1d(width)
        self.fc_final = nn.Linear(width, output_dim) 

    def forward(self, x):
        x = self.dense_block1(x)
        x = self.identity_block1(x)
        x = self.identity_block2(x)

        x = self.dense_block2(x)
        x = self.identity_block3(x)
        x = self.identity_block4(x)

        x = self.dense_block3(x)
        x = self.identity_block5(x)
        x = self.identity_block6(x)

        x = self.final_bn(x)
        x = self.fc_final(x)

        return x


class ResNet50RegressionNet(skorch.NeuralNetRegressor):
    def score(self, x, y):
        y_pred = self.predict(x)
        mape = - mean_absolute_percentage_error(y, y_pred)
        self.history.record('mape', mape.item())
        return mape
    

# Example usage:
input_dim = 100
output_dim = 13
num_samples = 20000

x_data = np.random.randn(num_samples, input_dim)

# Define a non-linear relationship to generate labels (y)
def generate_y(x):
    y = np.zeros((x.shape[0], output_dim))
    for i in range(output_dim):
        y[:, i] = (
            np.sin(x[:, i % input_dim]) +
            np.cos(x[:, (i + 1) % input_dim]) +
            np.tan(x[:, (i + 2) % input_dim]) +
            np.power(x[:, (i + 3) % input_dim], 2) +
            0.1 * np.random.randn(x.shape[0])  # Add some noise
        )
    return y

y_data = generate_y(x_data)

scaler_x = StandardScaler()
x_data = scaler_x.fit_transform(x_data)

scaler_y = StandardScaler()
y_data = scaler_y.fit_transform(y_data)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


net = ResNet50RegressionNet(
    module=ResNet50Regression,
    module__input_dim=input_dim,
    module__output_dim=output_dim,
    optimizer__weight_decay=1,
    criterion=nn.MSELoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=0.0001,
    max_epochs=1000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks = [('mape_scoring_train', EpochScoring(scoring=mape_train_score, lower_is_better=False, on_train=True)),
                 ('mape_scoring_test', EpochScoring(scoring=mape_test_score, lower_is_better=False, on_train=False)),
                 ('r2_score_train', EpochScoring(scoring=r2_train_score, lower_is_better=False, on_train=True)),
                 ('r2_score_test', EpochScoring(scoring=r2_test_score, lower_is_better=False, on_train=False)),
                 ('mae_score_train', EpochScoring(scoring=mae_train_score, lower_is_better=True, on_train=True)),
                 ('mae_score_test', EpochScoring(scoring=mae_test_score, lower_is_better=True, on_train=False)),
                 ('early_stopper', EarlyStopping(monitor='train_loss', patience=100, threshold=0.0001)),
                 ('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,mode='min', factor=0.1, patience=50))
                 ]
)



# common_params = {
#     "X": x_train_tensor,
#     "y": y_train_tensor,
#     "train_sizes": np.linspace(0.1, 1.0, 3),
#     "scoring": "r2",
#     "score_type": "both",  # training and test scores
#     "n_jobs": -1,
#     "line_kw": {"marker": "o"},  # Use 'o' marker for plot lines
#     "std_display_style": "fill_between",  # Fill area for standard deviation
#     "score_name": "Accuracy",  # Name for the score metric
# }
# fig, ax = plt.subplots(figsize=(8, 6))
# LearningCurveDisplay.from_estimator(
#     net, **common_params, ax=ax
# )
# handles, label = ax.get_legend_handles_labels()
# ax.legend(handles[:2], ["Training Score", "Test Score"])
# # Add legend
# ax.legend()
# plt.savefig("/home/mind/jrobador/from_scratch/learning_curve.png")
# plt.close()


net.fit(x_train_tensor,y_train_tensor)# 
mae_test_sc     = net.history[:, 'mae_test_score']
mae_train_sc    = net.history[:, 'mae_train_score']
mape_test_sc    = net.history[:,'mape_test_score']
mape_train_sc   = net.history[:,'mape_train_score']
r2_test_sc      = net.history[:,'r2_test_score'] 
r2_train_sc     = net.history[:,'r2_train_score']
train_losses    = net.history[:,'train_loss']    
valid_losses    = net.history[:,'valid_loss']    # 
epochs = range(1, len(mae_test_sc)+1)# 
metrics = {
    'mae_test_score': mae_test_sc,
    'mae_train_score': mae_train_sc,
    'mape_test_score': mape_test_sc,
    'mape_train_score': mape_train_sc,
    'r2_test_score': r2_test_sc,
    'r2_train_score': r2_train_sc,
    'train_loss': train_losses,
    'valid_loss': valid_losses
}# 
output_dir = "/home/mind/jrobador/from_scratch"
os.makedirs(output_dir, exist_ok=True)# 
for metric_name, metric_values in metrics.items():
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metric_values, label=f'{metric_name}')
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    file_name = f'{metric_name}.png'
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()# 
 
scores = {
    "mae_test_score":   min(mae_test_sc  ),
    "mae_train_score":  min(mae_train_sc ),
    "mape_test_score":  max(mape_test_sc ),
    "mape_train_score": max(mape_train_sc),
    "r2_test_score":    max(r2_test_sc   ),
    "r2_train_score":   max(r2_train_sc  ),
    "train_loss":       min(train_losses ),
    "valid_loss":       min(valid_losses )
}# 

best_scores_file = "model_best_scores.txt"# 
with open(os.path.join(output_dir, best_scores_file), 'w') as file:
    for key, value in scores.items():
        file.write(f"{key}: {value}\n")
