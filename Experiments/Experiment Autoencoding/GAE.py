# Will hold the code for the model code, will move to here in the future
from torch_geometric.datasets import QM9
import torch
from torch_geometric.nn import GAE, GCNConv

# define the encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# define the GAE model
def build_model(encoder_in, encoder_out):
    model = GAE(GCNEncoder(encoder_in, encoder_out))
    return model