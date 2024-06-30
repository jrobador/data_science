import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize the feature vectors
        z_i_normalized = F.normalize(z_i, dim=-1, p=2)
        z_j_normalized = F.normalize(z_j, dim=-1, p=2)

        # Compute the similarity scores
        sim_ij = torch.matmul(z_i_normalized, z_j_normalized.T) / self.temperature
        sim_ji = torch.matmul(z_j_normalized, z_i_normalized.T) / self.temperature

        # Construct the positive and negative pairs
        positive_pairs = torch.cat([sim_ij, sim_ji], dim=1)
        negative_pairs = torch.cat([sim_ij, sim_ji], dim=0)

        # Calculate the log-likelihoods
        log_likelihoods = F.log_softmax(positive_pairs, dim=1)

        # Compute the loss
        loss = -torch.mean(torch.diag(log_likelihoods))

        return loss

# Example usage:
# Assuming z_i and z_j are your feature vectors (output from your neural network) for positive pairs
z_i = torch.randn(64, 128)  # Example shape: (batch_size, feature_dim)
z_j = torch.randn(64, 128)

# Initialize the NTXentLoss with a specified temperature
temperature = 0.5
nt_xent_loss = NTXentLoss(temperature)

# Calculate the loss
loss = nt_xent_loss(z_i, z_j)

# Backpropagate and update the model parameters
loss.backward()
# Further optimization steps...