import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import train_loader, val_loader 
from mlp import Similarity
from losses import ContrastiveLoss
from config import num_epochs, hidden_size, color_channel, image_size, vocab_size, temperature, log_interval, embedding_dim, label_map

import torch
import logging

logging.basicConfig(filename='training.log', level=logging.INFO)

model = Similarity(image_channels=color_channel, image_size=image_size, label_map=label_map,
                   hidden_dim=hidden_size, embedding_dim=embedding_dim, vocab_size=vocab_size)

contrastive_loss = ContrastiveLoss(temperature)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1) 

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader):
            images          = images.view(images.size(0), -1) 
            x_img, x_label  = model(images, labels)
            loss            = criterion(x_img, x_label)
            total_loss      += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss

def main():
    best_loss = float('inf')
    best_epoch = 0

    # Training routine
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.view(images.size(0), 3, 32, 32)

            x_img, x_label = model(images, labels)
    
            loss = contrastive_loss(x_img, x_label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {average_loss:.4f}')

        # Validation
        val_loss = evaluate(model, val_loader, contrastive_loss)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        scheduler.step() 

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')

    logging.info(f"Training complete. Best validation loss: {best_loss:.4f} at epoch {best_epoch+1}")

if __name__ == '__main__':
    main()