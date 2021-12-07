from torch_geometric.datasets import QM9
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import numpy as np
import math
from GAE import build_model
from Utils import plot_epochs_history

# loading the QM9 dataset
dataset = QM9(root="tmp/QM9")

# 130,831 graphs are in the QM9 dataset
num_of_graphs = len(dataset)
train_idx = math.floor(0.80 * num_of_graphs)

# shuffling the dataset
torch.manual_seed(7869)
dataset.shuffle()

# splitting into training and testing sets
train_dataset = dataset[:train_idx]
test_dataset = dataset[train_idx:]

# training and testing loaders
train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# defining some variables for training the model
num_features = dataset.num_features
encoder_out = 32
epochs = 10

# define the graph autoencoder
model = build_model(num_features, encoder_out)

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# training the model


def train():
    model.train()

    for data in train_loader:
        # speed up the computation
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        # back propagation
        loss.backward()
        optimizer.step()
        # clear gradients
        optimizer.zero_grad()


# testing the model, very similar to train, but the gradients don't update

def test(loader):
    model.eval()

    loss_history = []
    for data in loader:
        with torch.no_grad():
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            z = model.encode(x, edge_index)

            # predictions.append(o)
            loss = model.recon_loss(z, edge_index)
            loss_history.append(loss)
    return np.sum(loss_history) / len(loader.dataset)


train_loss_history = []
test_loss_history = []


# iterating over the epochs
for epoch in range(1, epochs+1):
    train()
    train_loss = test(train_loader)
    test_loss = test(test_loader)

    # store the train and test reconstruction loss so we can visualize later
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    print(f'Epoch: {epoch:03d}, Train Recon Loss: {train_loss:.4f}, Test Recon Loss: {test_loss:.4f}')


# visualization!!!
plot_epochs_history(train_loss_history, "Train Reconstruction Loss over Time", "Epochs", "Reconstruction Loss")
plot_epochs_history(train_loss_history, "Test Reconstruction Loss over Time", "Epochs", "Reconstruction Loss")
