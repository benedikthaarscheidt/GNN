import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Metric
import torchmetrics
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
import numpy as np
import torch_scatter
from torch_scatter import scatter_add, scatter_max


class ModularPathwayConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='sum', pathway_groups=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.pathway_groups = pathway_groups  #
        # Adding nodes and neighbors features in a 1-d tensor
        self.mlp = nn.Sequential(
            nn.Linear(2*in_channels, out_channels),  # h_v + aggr(h_N(v))
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None, pathway_mode=False):
        x_updated=self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        if not pathway_mode or self.pathway_groups is None:
            # Global message passing without pathway mode
            return x_updated

        x_updated = x_updated.clone()  # Clone x to avoid overwriting the original tensor
        for pathway, nodes in self.pathway_groups.items():
    
            # Zero out the pathway nodes in `x` but keep the rest intact
            if pathway_mode:
                temp_x = x.clone()
                

            sub_edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=False)
    
            # Ensure sub_edge_index is only for valid edges involving the pathway
            valid_mask = torch.isin(sub_edge_index[0], nodes) & torch.isin(sub_edge_index[1], nodes)
    
            # Filter sub_edge_index using valid_mask
            sub_edge_index = sub_edge_index[:, valid_mask]

            # Handle edge attributes if available
            if edge_attr is not None:
                # Filter the original edge_attr based on the valid_mask for the original edge_index
                edge_mask = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
                sub_edge_attr = edge_attr[edge_mask]  
            else:
                sub_edge_attr = None  
    
            # Perform propagation for the pathway and update the pathway nodes only
            pathway_features = self.propagate(sub_edge_index, x=temp_x, edge_attr=sub_edge_attr)
            
            if pathway_features.dim() == 1 or pathway_features.shape[1] == 1:
                # Single feature case: Directly assign
                x_updated[nodes] = pathway_features[nodes]
            else:
                x_updated[nodes, :] = pathway_features[nodes, :] 
        print("next layer")
        return x_updated







    def propagate(self, edge_index, x,edge_attr=None, pathway_mode=False):
        """
        Custom propagation function to aggregate messages from neighbors using index_add_.
        Args:
            edge_index (Tensor): Edge indices [2, E].
            x (Tensor): Node features [N, 1] (one feature per node).
            edge_attr (Tensor, optional): Edge attributes [E].
            pathway_mode (bool): Whether pathway-specific aggregation is used.
        Returns:
            Tensor: Updated node features [N, out_channels] (N nodes, 2 features).
        """
        # Extract source (row) and target (col) nodes from edge_index
        row, col = edge_index[0], edge_index[1]

        # Ensure edge_index dimensions are correct
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        # Compute messages (these will be from source nodes to target nodes)
        messages = self.message(x[row], x[col], edge_attr)
        out = torch.zeros((x.size(0), messages.size(1)), device=x.device) 
    
        if self.aggr == 'mean' or self.aggr == 'sum':

            # Sum aggregation using index_add_ for target nodes (from col indices)
            out.index_add_(0, col, messages)
            
            if self.aggr == 'mean':
                # Normalize the aggregated messages by the degree (number of neighbors)
                degree = torch.bincount(col, minlength=x.size(0)).clamp(min=1)

                out = out / degree.unsqueeze(-1)  # Normalize by degree
                
        elif self.aggr == 'max':
            # Initialize the output tensor (same size as the node features)
            out = torch.zeros((x.size(0), messages.size(1)), device=x.device)
        
            # Compute the absolute values of the messages
            abs_messages = torch.abs(messages)
        
            # Iterate over all nodes (0 to x.size(0))
            for node in range(x.size(0)):

                mask = (col == node)
        
                if mask.any(): 

                    node_messages = messages[mask] 
                    
                    abs_node_messages = abs_messages[mask]
                    
                    # Find the index of the maximum absolute value for each feature
                    max_indices = torch.argmax(abs_node_messages, dim=0)  # Shape: [num_features]
        
                    # Retrieve the corresponding signed messages
                    max_messages = node_messages[max_indices, torch.arange(messages.size(1))]
                    # Assign the signed max messages to the output tensor
                    out[node] = max_messages
                    
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggr}")
        
        return out


    def message(self, x_i, x_j, edge_attr=None):
        """
        Compute messages for edges based on source and target node features,
        where the sending node's feature is scaled by the edge attribute,
        and both features are passed together to the MLP, then added along the second dimension.
        Args:
            x_i (Tensor): Features of source nodes (sending node) [E, in_channels].
            x_j (Tensor): Features of target nodes (receiving node) [E, in_channels].
            edge_attr (Tensor, optional): Attributes of edges [E].
        Returns:
            Tensor: Messages for edges [E, out_channels].
        """
        # Ensure x_i and x_j have the expected number of features
        if x_i.shape[1] != x_j.shape[1]:
            print("Source and target node features must have the same dimension.")
   
        # If edge_attr is provided, scale the sending node's feature (x_i) by the edge weight
        if edge_attr is not None:
            scaled_x_i = x_i * edge_attr.view(-1, 1)  # Scale the sending node feature (x_i)
        else:
            scaled_x_i = x_i  # No edge attribute, use the sending node feature directly
    
        # Concatenate the scaled sending node feature (x_i) and receiving node feature (x_j)
        combined_message = torch.cat([scaled_x_i, x_j], dim=1)  # Concatenate along the feature dimension
    
        # Pass the concatenated message through the MLP
        message = self.mlp(combined_message)  # Shape: [E, 2] after MLP transformation
        # Sum the two transformed features along the second dimension (feature dimension)
        #message = message.sum(dim=1, keepdim=True)  # Sum along dim=1 to get a single value per edge
        #this wont work as the mlp needs at least 2 points in space 
    
        return message


class ModularGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pathway_groups=None, layer_modes=None, pooling_mode='none', aggr_modes=None):
        """
        Args:
            input_dim (int): Number of input features per node.
            hidden_dim (int): Number of features in hidden layers.
            output_dim (int): Number of output features per node or graph.
            pathway_groups (dict, optional): Mapping of pathway names to lists of nodes.
            layer_modes (list, optional): Modes for each layer (True = pathway, False = global).
            pooling_mode (str): Pooling strategy ('none', 'scalar', 'pathway').
            aggr_modes (list, optional): Aggregation types for each layer ('sum', 'mean', 'max').
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.pathway_groups = pathway_groups
        self.pooling_mode = pooling_mode

        if layer_modes is None:
            layer_modes = [False] * 3  # Default to global convolution for all layers
        if aggr_modes is None:
            aggr_modes = ['sum'] * 3  # Default to sum aggregation for all layers
        assert len(layer_modes) == len(aggr_modes), "Layer modes and aggregation types must match in length."

        # Input layer
        self.layers.append(ModularPathwayConv(input_dim, hidden_dim, aggr=aggr_modes[0], pathway_groups=pathway_groups))

        # Hidden layers
        for mode, aggr in zip(layer_modes[1:-1], aggr_modes[1:-1]):
            self.layers.append(ModularPathwayConv(hidden_dim, hidden_dim, aggr=aggr, pathway_groups=pathway_groups))

        # Output layer
        self.layers.append(ModularPathwayConv(hidden_dim, output_dim, aggr=aggr_modes[-1], pathway_groups=pathway_groups))

        self.layer_modes = layer_modes

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the GNN.
        """
        # Process layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr, pathway_mode=self.layer_modes[i])

        # Handle different pooling modes
        if self.pooling_mode == 'none':
            return x  # No pooling, return per-node embeddings
        elif self.pooling_mode == 'scalar':
            return x.mean(dim=1, keepdim=True)  # Per-node scalar embedding
        elif self.pooling_mode == 'pathway':
            if self.pathway_groups is None:
                raise ValueError("Pathway groups must be provided for pathway-specific pooling.")
            return self.aggregate_by_pathway(x)  # Pathway-specific pooling
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def aggregate_by_pathway(self, x):
        """
        Aggregates embeddings for nodes within each pathway group.
        """
        pathway_embeddings = []
        for pathway, nodes in self.pathway_groups.items():
            # Extract features for nodes in the current pathway
            pathway_features = x[nodes]
            # Aggregate features within the pathway (e.g., mean pooling)
            pathway_embedding = torch.mean(pathway_features, dim=0, keepdim=True)
            pathway_embeddings.append(pathway_embedding)

        # Concatenate all pathway embeddings into a single matrix
        # Shape: (num_pathways, output_dim)
        return torch.cat(pathway_embeddings, dim=0)





