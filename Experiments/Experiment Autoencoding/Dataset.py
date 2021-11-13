from torch_geometric.datasets import QM9
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.utils import train_test_split_edges
import numpy as np
import math
from GAE import build_model

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
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# defining some variables for training the model
num_features = dataset.num_features
encoder_out = 32
epochs = 50

# define the graph autoencoder
model = build_model(num_features, encoder_out)

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# training the model
def train():
    model.train()
    for data in train_loader:
        # we keep track of all the reconstruction loss
        loss_history = []

        # speed up the computation
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss_history.append(loss)
        # back propagation
        loss.backward()
        optimizer.step()
        # clear gradients
        optimizer.zero_grad()
    # average reconstruction loss per epoch
    return np.sum(loss_history) / len(loss_history)


# iterating over the epochs
for epoch in range(1, epochs+1):
    l = train()
    print("Epoch {}: Reconstruction loss {}".format(epoch, l))