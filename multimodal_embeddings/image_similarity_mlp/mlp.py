import torch
import torch.nn as nn
import torch.nn.functional as F


# The goal of this architectures is to get the image and text features.
class ImageMLP(nn.Module):
    def __init__(self, image_channels, image_size, hidden_dim):
        super(ImageMLP, self).__init__()
        self.conv1  = nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1    = nn.Linear(64 * (image_size // 4) * (image_size // 4), hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_img):
        x_img      = F.relu(self.conv1(x_img))
        x_img      = self.pool(x_img)
        x_img      = F.relu(self.conv2(x_img))
        x_img      = self.pool(x_img)
        x_img      = x_img.view(x_img.size(0), -1)
        x_img      = F.relu(self.fc1(x_img))
        x_image    = self.fc2(x_img)
        return x_image
    
class TextMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_map):
        super(TextMLP, self).__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim)
        self.fc1        = nn.Linear(embedding_dim, hidden_dim)
        self.label_map  = label_map

    def forward(self, label):
        input_text      = self.label_map.get(label, "")
        words           = input_text.split()
        vocab           = {word: idx for idx, word in enumerate(words)}
        indices         = [vocab[word] for word in words if word in vocab]
        if not indices:
            return torch.zeros(1, self.fc1.out_features)
        x_lbl           = torch.tensor(indices, dtype=torch.long)
        x_lbl           = self.embedding(x_lbl)
        x_lbl           = torch.mean(x_lbl, dim=0, keepdim=True)
        x_label         = F.relu(self.fc1(x_lbl))

        return x_label

#Then, here we instantiate both MLP and normalize the features
class Similarity(nn.Module):
    def __init__(self, image_channels, image_size, hidden_dim, vocab_size, embedding_dim, label_map):
        super(Similarity, self).__init__()
        self.image_mlp = ImageMLP(image_channels, image_size, hidden_dim)   
        self.text_mlp  = TextMLP (vocab_size, embedding_dim, hidden_dim, label_map)

    def forward(self, x_img, x_lbl):
        x_image        = self.image_mlp(x_img)
        x_label        = self.text_mlp(x_lbl)
        
        # L2 normalize the feature representations
        x_image        = F.normalize(x_image, p=2, dim=1)
        x_label        = F.normalize(x_label, p=2, dim=1)

        return x_image, x_label
    
"""
image_encoder and text_encoder modules are used to extract feature representations 
from the input image and text. These feature representations are then normalized 
using L2 normalization to ensure that they have unit magnitude.
"""