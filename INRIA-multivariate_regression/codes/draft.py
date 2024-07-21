import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Model definition
class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = nn.Parameter(torch.randn(output_size, input_size))  # Random initialization of weights
        self.bias = nn.Parameter(torch.randn(output_size))  # Random initialization of bias

    def forward(self, x):
        # Forward pass of the linear layer. x: Input data, shape (batch_size, input_size)
        self.input = x
        self.output = torch.matmul(x, self.weights.t()) + self.bias
        return self.output

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[128, 64], output_dim=1):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(LinearLayer(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class RidgeRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, alpha=0.01):
        super(RidgeRegression, self).__init__()
        self.mlp = MLP(input_dim, hidden_dims, output_dim)
        self.alpha = alpha
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.mlp(x)

    def l2_regularization(self):
        l2_norm = 0
        for param in self.mlp.parameters():
            l2_norm += torch.norm(param)
        return self.alpha * l2_norm

X = 1 * np.random.rand(50, 1)
input_dim = X.shape[1]

y = 0.5 + 3 * X + np.random.randn(50, 1)
output_dim = y.shape[1]

inputs = torch.tensor(X, dtype=torch.float32)
targets = torch.tensor(y, dtype=torch.float32)


model = RidgeRegression(input_dim, hidden_dims=[256, 128], output_dim=output_dim)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = model.loss_function(outputs, targets) + model.l2_regularization()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
inputs_test = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    predictions = model(inputs_test).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X_test, predictions, color='red', label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Actual vs Predicted Data')
plt.legend()
plt.show()