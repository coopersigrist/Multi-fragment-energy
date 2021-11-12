from torch_geometric.datasets import QM9
import torch
from torch_geometric.nn import GAE, GCNConv
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.utils import train_test_split_edges
import math

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
# train_loader = DataLoader(train_dataset, batch_size=32)
# test_loader = DataLoader(test_dataset, batch_size=32)

'''implementations for the graph autoencoder
   first we define the encoder we are using '''

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# defining some variables for training the model
num_features = dataset.num_features
encoder_out = 2
epochs = 50

# define the graph autoencoder
model = GAE(GCNEncoder(num_features, encoder_out))

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# training the model
def train():
    model.train()
    for data in train_dataset:
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(float(loss))

train()