import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, aggr
from torch_geometric.utils import subgraph
from torch_scatter import scatter_add, scatter_max

class ModularPathwayConv(MessagePassing):
    def __init__(self, in_channels, out_channels,num_pathways_per_instance, aggr_mode="sum", pathway_groups=None):
        super().__init__(aggr=aggr_mode)  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pathway_groups = pathway_groups
        self.num_pathways_per_instance=num_pathways_per_instance
        self.linear = torch.nn.Linear(in_channels, out_channels)

        print(self.num_pathways_per_instance)
        if isinstance(self.aggr, str):
            print(f"Initialized with '{self.aggr}' string-based aggregator")
        else:
            print(f"Initialized with {type(self.aggr).__name__} aggregator")

    def forward(self, x, edge_index, edge_attr=None, pathway_mode=False, pathway_tensor=None, batch=None):
        if batch is None:
            raise ValueError("The 'batch' parameter is required for batched processing but is None.")

        x = x.float()
        #x = self.linear(x)

        if (not pathway_mode) or (pathway_tensor is None and self.pathway_groups is None):
            x_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            
        else:
            x_updated = self._process_pathways(x, edge_index, edge_attr, pathway_tensor, batch)

        return x_updated


    def _process_pathways(self, x, edge_index, edge_attr, pathway_tensors, batch):
        x_updated = torch.zeros_like(x)  # Initialize the updated features
    
        # **Compute node offsets** for each graph in the batch
        node_offsets = torch.cumsum(batch.bincount(), dim=0)
        node_start_indices = torch.cat([torch.tensor([0]), node_offsets[:-1]])  # Start index of each graph's nodes
        
        if self.pathway_groups is not None:
            # **Global Pathway Mode**
            num_graphs = batch.unique().size(0)
            
            # Repeat pathway_groups for every graph in the batch
            graph_idx = torch.arange(num_graphs).repeat_interleave(self.num_pathways_per_instance)
            
            # Shift pathway indices globally for each graph in the batch
            pathway_groups_shifted = self.pathway_groups + node_start_indices[graph_idx].view(-1, 1)
        else:
            # **Individual Pathway Mode**
            # Here we use the pathway_tensors for each graph separately
            num_graphs = batch.unique().size(0)
            
            pathway_groups_shifted = []
            for graph_idx in range(num_graphs):
                start_idx = graph_idx * self.num_pathways_per_instance
                end_idx = start_idx + self.num_pathways_per_instance
                local_pathway_tensor = pathway_tensors[start_idx:end_idx, :]
    
                node_offset = node_start_indices[graph_idx]
                
                # **Shift only the valid node indices, not -1**
                shifted_pathway_tensor = local_pathway_tensor.clone()
                shifted_pathway_tensor[shifted_pathway_tensor >= 0] += node_offset
                
                pathway_groups_shifted.append(shifted_pathway_tensor)
            
            pathway_groups_shifted = torch.cat(pathway_groups_shifted, dim=0)
        print(f"shifted pathway_groups:{pathway_groups_shifted}")
        # **Pathway-wise propagation logic**
        for pathway_index in range(pathway_groups_shifted.size(0)):
            nodes = pathway_groups_shifted[pathway_index, :]
            
            nodes = nodes[nodes >= 0] 
            if nodes.numel() == 0:
                continue
    
            # **Subgraph for the pathway (batch-wide)**
            sub_edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=False)

            sub_edge_attr = None
            if edge_attr is not None:
                edge_mask = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
                sub_edge_attr = edge_attr[edge_mask]
    
            x_propagated = self.propagate(sub_edge_index, x=x, edge_attr=sub_edge_attr)
            # Update only the features for nodes in the pathway
            x_updated[nodes] += x_propagated[nodes] + x[nodes]
        print(f"x_updated[nodes]:{x_updated}")
        return x_updated
           


    def message(self, x_j, edge_attr=None):
        if edge_attr is not None: 
            return x_j * edge_attr.view(-1, 1)
        return x_j


class ModularGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_pathways_per_instance, pathway_groups=None, layer_modes=None, pooling_mode=None, aggr_modes=None):
        super().__init__()
        self.num_pathways_per_instance = num_pathways_per_instance
        self.layers = nn.ModuleList()
        self.pathway_groups = pathway_groups  
        self.pooling_mode = pooling_mode
        self.layer_modes = layer_modes or [False] * 3  
        

        if aggr_modes is None:
            aggr_modes = ['sum'] * 3  

        self.layers.append(ModularPathwayConv(input_dim, hidden_dim,num_pathways_per_instance, aggr_mode=aggr_modes[0], pathway_groups=pathway_groups))

        for mode, aggr_mode in zip(self.layer_modes[1:-1], aggr_modes[1:-1]):
            self.layers.append(ModularPathwayConv(hidden_dim, hidden_dim,num_pathways_per_instance, aggr_mode=aggr_mode, pathway_groups=pathway_groups))
        
        self.layers.append(ModularPathwayConv(hidden_dim, output_dim,num_pathways_per_instance, aggr_mode=aggr_modes[-1], pathway_groups=pathway_groups))

    def forward(self, x, edge_index, edge_attr=None, pathway_tensor=None, batch=None):
        # Determine batch size
        if batch is not None:
            batch_size = batch.unique().size(0)
        for i, layer in enumerate(self.layers):
            current_pathway_tensor = pathway_tensor if pathway_tensor is not None else self.pathway_groups
            x = layer(x, edge_index, edge_attr=edge_attr, pathway_mode=self.layer_modes[i], pathway_tensor=current_pathway_tensor,batch=batch)
    
        # Pooling logic
        # Aggregate pathway embeddings
        if self.pooling_mode == 'pathway':
            pathway_source = pathway_tensor if pathway_tensor is not None else self.pathway_groups
            if pathway_source is not None:
                x = self.aggregate_by_pathway(x, edge_index, edge_attr, pathway_source, batch=batch)

    
        elif self.pooling_mode == 'scalar':
            from torch_geometric.nn import global_mean_pool
            if batch is None:
                raise ValueError("Batch tensor is required for global pooling with multiple graphs")
            x = global_mean_pool(x, batch)
            x = x.mean(dim=1, keepdim=True)
    
        elif self.pooling_mode is None:
            from torch_geometric.nn import global_mean_pool
            if batch is None:
                raise ValueError("Batch tensor is required to flatten node embeddings per graph")
            x = global_mean_pool(x, batch)
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
    
        else:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")
    
        return x

    def aggregate_by_pathway(self, x, edge_index, edge_attr, pathway_tensors, batch):

        print("pooling it")
        # **Compute node offsets** for each graph in the batch
        node_offsets = torch.cumsum(batch.bincount(), dim=0)
        node_start_indices = torch.cat([torch.tensor([0]), node_offsets[:-1]])  # Start index of each graph's nodes
        
        if self.pathway_groups is not None:
            # **Global Pathway Mode**
            num_graphs = batch.unique().size(0)
            
            # Repeat pathway_groups for every graph in the batch
            graph_idx = torch.arange(num_graphs).repeat_interleave(self.num_pathways_per_instance)
            
            # Shift pathway indices globally for each graph in the batch
            pathway_groups_shifted = self.pathway_groups.clone()
            pathway_groups_shifted[pathway_groups_shifted >= 0] += node_start_indices[graph_idx].view(-1, 1)
        else:
            # **Individual Pathway Mode**
            num_graphs = batch.unique().size(0)
            
            pathway_groups_shifted = []
            for graph_idx in range(num_graphs):
                start_idx = graph_idx * self.num_pathways_per_instance
                end_idx = start_idx + self.num_pathways_per_instance
                local_pathway_tensor = pathway_tensors[start_idx:end_idx, :]
    
                node_offset = node_start_indices[graph_idx]
                
                # **Shift only the valid node indices, not -1**
                shifted_pathway_tensor = local_pathway_tensor.clone()
                shifted_pathway_tensor[shifted_pathway_tensor >= 0] += node_offset
                
                pathway_groups_shifted.append(shifted_pathway_tensor)
            
            pathway_groups_shifted = torch.cat(pathway_groups_shifted, dim=0)
    
        batch_size = batch.max().item() + 1
        embedding_dim = x.size(1)
        
        # Store aggregated pathway embeddings
        aggregated_pathway_features = torch.zeros(
            (batch_size, self.num_pathways_per_instance, embedding_dim), device=x.device
        )
        
    
        # **Pathway-wise aggregation logic**
        for pathway_index in range(pathway_groups_shifted.size(0)):
            nodes = pathway_groups_shifted[pathway_index, :]
            nodes = nodes[nodes >= 0]  # Remove padding (-1)
            print(f"processing nodes:{nodes}")
            if nodes.numel() == 0:
                continue
            
            # **Aggregate features for nodes in this pathway**
            pathway_features = x[nodes]
            
            if pathway_features.size(0) > 0:
                pathway_embedding = torch.mean(pathway_features, dim=0)
                
                # Identify which graph this pathway belongs to
                graph_idx = pathway_index // self.num_pathways_per_instance
                pathway_idx = pathway_index % self.num_pathways_per_instance
                
                # Store the result in the final tensor
                aggregated_pathway_features[graph_idx, pathway_idx] = pathway_embedding
        print(f"aggregated pathway features:{aggregated_pathway_features}")
        return aggregated_pathway_features