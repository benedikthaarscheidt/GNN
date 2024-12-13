import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, aggr
from torch_geometric.utils import subgraph
from torch_scatter import scatter_add, scatter_max

class ModularPathwayConv(MessagePassing):
    def __init__(self, in_channels, out_channels,num_pathways_per_instance, aggr_mode="sum"):
        super().__init__(aggr=aggr_mode)  
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.num_pathways_per_instance=num_pathways_per_instance
        self.linear = torch.nn.Linear(in_channels, out_channels)

        if isinstance(self.aggr, str):
            print(f"Initialized with '{self.aggr}' string-based aggregator")
        else:
            print(f"Initialized with {type(self.aggr).__name__} aggregator")

    def forward(self, x, edge_index, edge_attr=None, pathway_mode=False, pathway_tensor=None, batch=None):
        if batch is None:
            raise ValueError("The 'batch' parameter is required for batched processing but is None.")

        x = x.float()
        x = self.linear(x)

        if not pathway_mode or pathway_tensor is None:

            x_updated = (self.propagate(edge_index, x=x, edge_attr=edge_attr) + x)#/2
        else:
            x_updated = self._process_pathways(x, edge_index, edge_attr, pathway_tensor, batch)

        return x_updated


    def _process_pathways(self, x, edge_index, edge_attr, pathway_tensors, batch):
        x_updated = torch.zeros_like(x)  
            
        for pathway_index in range(pathway_tensors.size(0)):
            nodes = pathway_tensors[pathway_index, :]
            nodes = nodes[nodes >= 0] 
            if nodes.numel() == 0:
                continue
    
            sub_edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=False)
    
            sub_edge_attr = None
            if edge_attr is not None:
                edge_mask = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
                sub_edge_attr = edge_attr[edge_mask]
    
            x_propagated = self.propagate(sub_edge_index, x=x, edge_attr=sub_edge_attr)

            x_updated[nodes] += (x_propagated[nodes] + x[nodes])#/2
            
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

        self.layers.append(ModularPathwayConv(input_dim, hidden_dim,num_pathways_per_instance, aggr_mode=aggr_modes[0]))

        for mode, aggr_mode in zip(self.layer_modes[1:-1], aggr_modes[1:-1]):
            self.layers.append(ModularPathwayConv(hidden_dim, hidden_dim,num_pathways_per_instance, aggr_mode=aggr_mode))
        
        self.layers.append(ModularPathwayConv(hidden_dim, output_dim,num_pathways_per_instance, aggr_mode=aggr_modes[-1]))

    
    def _shift_global_pathways(self, batch):
        
        node_offsets = torch.cumsum(batch.bincount(), dim=0)
        node_start_indices = torch.cat([torch.tensor([0], device=node_offsets.device), node_offsets[:-1]]) 
        
        num_graphs = batch.unique().size(0)
        
        pathway_groups_shifted = self.pathway_groups.repeat(num_graphs, 1)
        
        pathway_tensors_reshaped = pathway_groups_shifted.view(num_graphs, self.pathway_groups.size(0), -1)
        
        for graph_idx in range(num_graphs):
            node_offset = node_start_indices[graph_idx]
            shifted_pathway_tensor = pathway_tensors_reshaped[graph_idx]
            
            shifted_pathway_tensor[shifted_pathway_tensor >= 0] += node_offset
        
        return pathway_tensors_reshaped.view(-1, pathway_tensors_reshaped.size(-1))
        
    
    def _shift_individual_pathways(self, pathway_tensors, batch):

        node_offsets = torch.cumsum(batch.bincount(), dim=0)
        node_start_indices = torch.cat([torch.tensor([0], device=node_offsets.device), node_offsets[:-1]])  # Start index for each graph
        
        num_graphs = batch.unique().size(0)
        pathway_groups_shifted = pathway_tensors.clone()
        
        pathway_tensors_reshaped = pathway_tensors.view(num_graphs, self.num_pathways_per_instance, -1)
        
        for graph_idx in range(num_graphs):
            node_offset = node_start_indices[graph_idx]
            shifted_pathway_tensor = pathway_tensors_reshaped[graph_idx]
            
            shifted_pathway_tensor[shifted_pathway_tensor >= 0] += node_offset
        
        return pathway_tensors_reshaped.view(-1, pathway_tensors.size(-1))

    
    def forward(self, x, edge_index, edge_attr=None, pathway_tensor=None, batch=None):
        
        if pathway_tensor is not None:
            pathway_groups_shifted = self._shift_individual_pathways(pathway_tensor, batch)
        elif self.pathway_groups is not None:
            pathway_groups_shifted = self._shift_global_pathways(batch)
        else:
            pathway_groups_shifted = None
            
        if batch is not None:
            batch_size = batch.unique().size(0)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr, pathway_mode=self.layer_modes[i], pathway_tensor=pathway_groups_shifted,batch=batch)
    

        if self.pooling_mode == 'pathway':
            if pathway_groups_shifted is not None:
                x = self.aggregate_by_pathway(x, edge_index, edge_attr, pathway_groups_shifted, batch=batch)
            else:
                print("pathway tensor required for pathway specific pooling")
                from torch_geometric.nn import global_mean_pool
                if batch is None:
                    raise ValueError("Batch tensor is required for global pooling with multiple graphs")
                x = global_mean_pool(x, batch)
                x = x.mean(dim=1, keepdim=True)
    
    
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
        batch_size = batch.max().item() + 1
        embedding_dim = x.size(1)

        aggregated_pathway_features = torch.zeros(
            (batch_size, self.num_pathways_per_instance, embedding_dim), device=x.device
        )
        
        for pathway_index in range(pathway_tensors.size(0)):
            nodes = pathway_tensors[pathway_index, :]
            nodes = nodes[nodes >= 0]  # Remove padding (-1)

            if nodes.numel() == 0:
                continue
            
            pathway_features = x[nodes]
            
            if pathway_features.size(0) > 0:
                pathway_embedding = torch.mean(pathway_features, dim=0)
                
                graph_idx = pathway_index // self.num_pathways_per_instance
                pathway_idx = pathway_index % self.num_pathways_per_instance
                
                aggregated_pathway_features[graph_idx, pathway_idx] = pathway_embedding
        
        return aggregated_pathway_features