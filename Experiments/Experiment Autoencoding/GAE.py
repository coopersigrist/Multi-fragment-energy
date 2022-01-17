# Will hold the code for the model code, will move to here in the future
import torch
from torch_geometric.nn import GAE, GCNConv

# define the encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels,  out_channels * 2)
        self.conv2 = GCNConv(out_channels * 2, out_channels * 4)
        self.conv3 = GCNConv(out_channels * 4, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)

# define the GAE model
def build_model(encoder_in, encoder_out):
    # should use the default decoder provided by pytorch geometric - innerproduct decoder
    model = GAE(GCNEncoder(encoder_in, encoder_out), None)
    return model