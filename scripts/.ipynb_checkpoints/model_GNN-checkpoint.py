import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, aggr
from torch_geometric.utils import subgraph
from torch_scatter import scatter_add, scatter_max,scatter_softmax,scatter
import torch.autograd.profiler as profiler
from sklearn.linear_model import Ridge
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class ModularPathwayConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_pathways_per_instance, aggr_mode="sum"):
        super().__init__(aggr=aggr_mode)  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_pathways_per_instance = num_pathways_per_instance
        self.linear = torch.nn.Linear(in_channels, out_channels)
        #self.batch_norm = nn.BatchNorm1d(out_channels)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        #self.dropout = nn.Dropout(p=0.1) 
        self.buffer = None
        self.edge_attention = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x, edge_index, edge_attr=None, pathway_mode=True, pathway_subgraphs=None, batch=None):
        torch.cuda.empty_cache()
        
        if batch is None:
            raise ValueError("The 'batch' parameter is required for batched processing but is None.")
        
        # Apply node transformations
        x = x.float()
        x = self.linear(x)
        #x = self.batch_norm(x)
        #x = self.dropout(x)



        if pathway_mode and pathway_subgraphs:
            return self._process_precomputed_pathways(x, pathway_subgraphs)
        else:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr).add_(x)


    def _process_precomputed_pathways(self, x, pathway_subgraphs, batch_size=32):
        try:
            buffer = torch.zeros_like(x, device=x.device)
            

            pathway_batches = [pathway_subgraphs[i:i + batch_size] for i in range(0, len(pathway_subgraphs), batch_size)]
            
            for pathway_batch in pathway_batches:

                batch_nodes = torch.cat([nodes for _, _, nodes in pathway_batch], dim=0)
                batch_edge_index = torch.cat([sub_edge_index for sub_edge_index, _, _ in pathway_batch], dim=1)
                batch_edge_attr = None
                if any(sub_edge_attr is not None for _, sub_edge_attr, _ in pathway_batch):
                    batch_edge_attr = torch.cat(
                        [sub_edge_attr for _, sub_edge_attr, _ in pathway_batch if sub_edge_attr is not None], dim=0
                    )
                

                batch_out = self.propagate(batch_edge_index, x=x, edge_attr=batch_edge_attr)
                

                for (_, _, nodes) in pathway_batch:
                    buffer[nodes] += batch_out[nodes]
            
            return buffer
        except Exception as e:
            print("_process_precomputed_pathways")



    
#   def _process_precomputed_pathways(self, x, pathway_subgraphs):
#       if not hasattr(self, "buffer") or self.buffer is None or self.buffer.size() != x.size():
#           self.buffer = torch.zeros_like(x, device=x.device)
#       
#       self.buffer.zero_()  
#   
#       for sub_edge_index, sub_edge_attr, nodes in pathway_subgraphs:
#           if nodes.numel() == 0:
#               continue
#           self.buffer[nodes].add_(self.propagate(sub_edge_index, x=x, edge_attr=sub_edge_attr)[nodes])#.add_(x[nodes])
#       return self.buffer


        
    def message(self, x_j, edge_attr=None, index=None):
        try:
            if edge_attr is not None:
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
                edge_attr = edge_attr * self.alpha
                attention_scores = scatter_softmax(self.edge_attention(edge_attr), index=index, dim=0)
                try:
                    return x_j * attention_scores
                except  Exception as e:
                    raise ValueError(f"message")
            else:
                return x_j
        except Exception as e:
            print("message")





######################################################################################

class ModularGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_pathways_per_instance, aggr_mode, dropout=0.1):
        super().__init__()
        #self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.conv = ModularPathwayConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_pathways_per_instance=num_pathways_per_instance,
            aggr_mode=aggr_mode
        )
        self.post_processing = nn.Sequential(
            nn.LayerNorm(out_channels),
            #nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout)
        )

  #  def forward(self, x, edge_index, edge_attr=None, pathway_subgraphs=None, batch=None,pathway_mode=True):
  #      try:
  #          h = self.conv(x, edge_index, edge_attr=edge_attr, pathway_mode=pathway_mode, pathway_subgraphs=pathway_subgraphs, batch=batch)
  #          h = self.post_processing(h)
  #          try:
  #              return h + self.residual_scale * x
  #          except Exception as e:
  #              raise ValueError(f"Error in residual connection: {e}. Shapes of h: {h.shape}, x: {x.shape}. Ensure they are compatible")
  #      except Exception as e:
  #          print("forward ModularGNNLayer")
    def forward(self, x, edge_index, edge_attr=None, pathway_subgraphs=None, batch=None,pathway_mode=True):
            x = self.conv(x, edge_index, edge_attr=edge_attr,pathway_mode=pathway_mode, pathway_subgraphs=pathway_subgraphs, batch=batch)
            x = self.post_processing(x)
            return x
##########################################################################################

class ModularGNN(nn.Module):
    precomputed_subgraphs = None
    def __init__(self, input_dim, hidden_dim, output_dim,num_pathways_per_instance, pathway_groups=None, layer_modes=None, pooling_mode=None, aggr_modes=None,num_layers=5,fixed_dim=64):
        super().__init__()
        self.num_pathways_per_instance = num_pathways_per_instance
        self.pathway_groups = pathway_groups  
        self.pooling_mode = pooling_mode
        self.layer_modes = layer_modes or [False] * 3  
        if aggr_modes is None:
            aggr_modes = ['sum'] * 3  
        #for pathway pooling
        self.fixed_dim = fixed_dim
        self.linear_layer = nn.Linear(input_dim, fixed_dim, bias=False)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim

            self.layers.append(
                ModularGNNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_pathways_per_instance=num_pathways_per_instance,
                    aggr_mode=aggr_modes[i],
                    dropout=0.2
                )
            )


    
    def forward(self, x, edge_index, edge_attr=None, pathway_tensor=None, batch=None, recompute_subgraphs=False, prune_edges=False):
        try:
            if self.precomputed_subgraphs is None and pathway_tensor is not None:
                self.precomputed_subgraphs = self.precompute_subgraphs(pathway_tensor, edge_index, edge_attr)
        except Exception as e:
            print("precomputation of pathways is faulty")
            raise e
        shifted_subgraphs = (
            self.shift_subgraphs(self.precomputed_subgraphs, batch)
            if self.precomputed_subgraphs is not None
            else None
        )

        try:
            for i, layer in enumerate(self.layers):
                x = layer(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    pathway_mode=self.layer_modes[i],
                    pathway_subgraphs=shifted_subgraphs,
                    batch=batch
                )
        except Exception as e:
            print("somewhere is forward")
            raise e

        ######## this is just for the pooling later
        if pathway_tensor is not None:
            pathway_groups_shifted = self._shift_individual_pathways(pathway_tensor, batch)
        elif self.pathway_groups is not None:
            pathway_groups_shifted = self._shift_global_pathways(batch)
        else:
            pathway_groups_shifted = None
    
        if batch is not None:
            batch_size = batch.unique().size(0)  
        ##############

        try:
            
            if self.pooling_mode == 'pathway':
                if pathway_groups_shifted is not None:
                    x = self.aggregate_by_pathway(x, edge_index, edge_attr, pathway_groups_shifted, batch=batch)
                else:
                    print("Pathway tensor required for pathway-specific pooling")
                    from torch_geometric.nn import global_mean_pool
                    if batch is None:
                        raise ValueError("Batch tensor is required for global pooling with multiple graphs")

                    x = global_mean_pool(x, batch)  # Output shape: [batch_size, output_dim]
        
            elif self.pooling_mode == 'scalar':
                from torch_geometric.nn import global_mean_pool
                if batch is None:
                    raise ValueError("Batch tensor is required for global pooling with multiple graphs")
                x = global_mean_pool(x, batch)  # Global mean pooling across graphs
        
            elif self.pooling_mode is None:
                if batch is None:
                    raise ValueError("Batch tensor is required to flatten node embeddings per graph")
                

                num_nodes_per_graph = torch.bincount(batch)  # Gives the count of nodes per graph
                if not torch.all(num_nodes_per_graph == num_nodes_per_graph[0]):
                    raise ValueError("All graphs must have the same number of nodes when pooling is 'None'")
                

                batch_size = batch.max().item() + 1  # Total number of graphs
                num_nodes = num_nodes_per_graph[0].item()  # Number of nodes per graph
                x = x.view(batch_size, num_nodes * x.size(-1))  # Concatenate node features for each graph
            
            
    
            else:
                raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")
            #print(f"x shape after flattening:{x.shape}")
            return x
        except Exception as e:
            print("error in pooling")
            raise e 
            
    def aggregate_by_pathway(self, x, edge_index, edge_attr, pathway_tensors, batch):
        try:
            batch_size = batch.max().item() + 1
            valid_mask = pathway_tensors >= 0
            flat_pathway_indices = pathway_tensors[valid_mask] 
            flat_pathway_assignments = torch.arange(
                pathway_tensors.size(0), device=x.device
            ).repeat_interleave(valid_mask.sum(dim=1))
            flat_node_embeddings = x[flat_pathway_indices]
            try:
                transformed_features = self.linear_layer(flat_node_embeddings)
            except Exception as e:
                print("linear layer in aggregate_by_pathway")
                raise e 
            aggregated_pathways = torch.zeros(
                (pathway_tensors.size(0), self.fixed_dim), device=x.device, dtype=transformed_features.dtype
            )
            aggregated_pathways.index_add_(
                0, flat_pathway_assignments, transformed_features
            )
            pathway_counts = valid_mask.sum(dim=1, keepdim=True).to(x.device).float()
            aggregated_pathways /= pathway_counts
            aggregated_pathway_features = aggregated_pathways.view(
                batch_size, self.num_pathways_per_instance, self.fixed_dim
            )
            aggregated_pathway_features = aggregated_pathway_features.view(
                batch_size, -1
            )  
            return aggregated_pathway_features
        except Exception as e:
            print("error in aggregate_by_pathway")
            raise e
            
    def precompute_subgraphs(self, pathway_tensor, edge_index, edge_attr=None):
        precomputed_subgraphs = []
        for pathway in pathway_tensor:
            nodes = pathway[pathway >= 0] +
            sub_edge_index, edge_mask = subgraph(nodes, edge_index, relabel_nodes=False)
            sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
            precomputed_subgraphs.append((sub_edge_index, sub_edge_attr, nodes))
        return precomputed_subgraphs

    def shift_subgraphs(self, precomputed_subgraphs, batch):
        node_offsets = torch.cumsum(torch.bincount(batch), dim=0).to(batch.device)
        node_start_indices = torch.cat([torch.tensor([0], device=batch.device), node_offsets[:-1]])
    
        shifted_subgraphs = []
        for graph_idx in range(batch.max().item() + 1):
            graph_subgraphs = []
            for sub_edge_index, sub_edge_attr, nodes in precomputed_subgraphs:

                shifted_nodes = nodes + node_start_indices[graph_idx]
                shifted_edge_index = sub_edge_index.clone()
                shifted_edge_index[0] += node_start_indices[graph_idx]
                shifted_edge_index[1] += node_start_indices[graph_idx]
                graph_subgraphs.append((shifted_edge_index, sub_edge_attr, shifted_nodes))
            shifted_subgraphs.append(graph_subgraphs)
        return shifted_subgraphs
    
    def _shift_individual_pathways(self, pathway_tensors, batch):
        node_offsets = torch.cumsum(batch.bincount(), dim=0).to(batch.device)
        node_start_indices = torch.cat([torch.tensor([0], device=batch.device), node_offsets[:-1]])
        
        pathway_tensors_reshaped = pathway_tensors.view(batch.max() + 1, self.num_pathways_per_instance, -1).to(batch.device)
        
        for graph_idx in range(batch.max().item() + 1):
            node_offset = node_start_indices[graph_idx]
            shifted_pathway_tensor = pathway_tensors_reshaped[graph_idx]
            shifted_pathway_tensor[shifted_pathway_tensor >= 0] += node_offset  # No device mismatch
        
        return pathway_tensors_reshaped.view(-1, pathway_tensors.size(-1))

    def _shift_global_pathways(self, batch):
        node_offsets = torch.cumsum(batch.bincount(), dim=0).to(batch.device)
        node_start_indices = torch.cat([torch.tensor([0], device=node_offsets.device), node_offsets[:-1]]) 
        
        num_graphs = batch.unique().size(0)
        pathway_groups_shifted = self.pathway_groups.repeat(num_graphs, 1).to(batch.device)
        
        pathway_tensors_reshaped = pathway_groups_shifted.view(num_graphs, self.pathway_groups.size(0), -1)
        
        for graph_idx in range(num_graphs):
            node_offset = node_start_indices[graph_idx]
            shifted_pathway_tensor = pathway_tensors_reshaped[graph_idx]
            shifted_pathway_tensor[shifted_pathway_tensor >= 0] += node_offset  # No device mismatch now
        
        return pathway_tensors_reshaped.view(-1, pathway_tensors_reshaped.size(-1))


    def summary(self):
        print("ModularGNN Architecture:")
        print(f"Number of Layers: {len(self.layers)}")
        print(f"Pooling Mode: {self.pooling_mode}")
        print(f"Layer Modes: {self.layer_modes}")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i}: {layer}")
        