from torch_geometric.datasets import QM9
from torch_geometric.utils import train_test_split_edges
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE

# loading the QM9 dataset
dataset = QM9(root="tmp/QM9")

# 130,831 graphs are in the QM9 dataset
num_of_graphs = len(dataset)
print("Number of graphs in QM9:", num_of_graphs)

# defining some variables for training the model
num_features = dataset.num_features
epochs = 50

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


# define the graph autoencoder
model = GAE(GCNEncoder(num_features, num_features))