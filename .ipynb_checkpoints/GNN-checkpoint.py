import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_type='GCN'):
        super(GraphNeuralNetwork, self).__init__()
        
        # Choose the convolutional layer type
        if conv_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif conv_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
        
        # Linear layer for the output
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # First GNN layer + ReLU activation
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Second GNN layer + ReLU activation
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Pooling layer to get a graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)  # or global_max_pool
        
        # Linear layer for prediction
        x = self.lin(x)
        
        return x

# Example instantiation
input_dim = 1  # Adjust according to your node feature size
hidden_dim = 64  # Number of hidden units
output_dim = 1  # Output size (e.g., regression or classification)
gnn_model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)