from torch_geometric.datasets import QM9
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import numpy as np
import math
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from GAE import build_model
from Utils import plot_epochs_history
from early_stopping import EarlyStopping

# loading the QM9 dataset
dataset = QM9(root="tmp/QM9")

# 130,831 graphs are in the QM9 dataset
num_of_graphs = len(dataset)
train_idx = math.floor(0.80 * num_of_graphs)

# shuffling the dataset
torch.manual_seed(123)
dataset = dataset.shuffle()

# splitting into training and testing sets
train_dataset = dataset[:train_idx]
test_dataset = dataset[train_idx:]

# training and testing loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# defining some variables for training the model
num_features = dataset.num_features

epochs = 25

# define the graph autoencoder
model = build_model(num_features, num_features)

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.01)

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


# plot training and testing loss over the epochs
train_loss_history = []
test_loss_history = []

# define the early stop
es = EarlyStopping(patience=3)

# iterating over the epochs
for epoch in range(1, epochs+1):
    train()
    train_loss = test(train_loader)
    test_loss = test(test_loader)

    # if the model is no longer improving, stop the training
    if es.step(train_loss):
        print("Early Stop")
        break

    # store the train and test reconstruction loss so we can visualize later
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    print(f'Epoch: {epoch:03d}, Train Recon Loss: {train_loss:.4f}, Test Recon Loss: {test_loss:.4f}')


# visualization!!!
plot_epochs_history(train_loss_history, "Train Reconstruction Loss over Time", "Epochs", "Reconstruction Loss")
plot_epochs_history(test_loss_history, "Test Reconstruction Loss over Time", "Epochs", "Reconstruction Loss")

# seeing output info, since there seems a bug
model.eval()
# extracting graphs from the batched large graph
for data in test_loader:
    with torch.no_grad():
        for i in range(3):
            # original graph info
            graph = data.get_example(i)
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)

            # computing latent representation (nodes) and adjacent matrix
            z = model.encode(x, edge_index)
            # prob_edges = model.decode(z, edge_index)
            adj = model.decoder.forward_all(z, sigmoid=True)
            # sparsify the adjacent matrix into shape [2, num_nodes]
            o_edge_index = (adj > 0.8).nonzero(as_tuple=False).t()
            output = Data(x=z, edge_index=o_edge_index)
            print("Graph", i)

            g = to_networkx(graph, to_undirected=True)
            nx.draw(g)
            plt.show()

            n = to_networkx(output, to_undirected=True)
            nx.draw(n)
            plt.show()

            # z = model.encode(graph.x, graph.edge_index)
            # edge_prob = model.decode(z, graph.edge_index)
        break


