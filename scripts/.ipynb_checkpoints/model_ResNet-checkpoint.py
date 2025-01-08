import torch
from torch import nn
from torch import Tensor
from torchmetrics import Metric
import torchmetrics
from scripts import *


class ResNet(nn.Module):
    def __init__(self, embed_dim=44, hidden_dim=128, n_layers=6, dropout=0.1, norm="layernorm"):
        super().__init__()
        
        norm_choices = {"layernorm": nn.LayerNorm, "batchnorm": nn.BatchNorm1d, "identity": nn.Identity}
        norm_layer = norm_choices.get(norm, nn.Identity)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim)  
                )
            )
        
        self.final_layer = nn.Linear(embed_dim, 1) 
        self._init_weights()  

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0) 

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (layer(x) + x) * 0.5  
        x = self.final_layer(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, gnn, drug_mlp, resnet):

        super().__init__()
        self.gnn = gnn
        self.drug_mlp = drug_mlp
        self.resnet = resnet
        self.cell_weight = nn.Parameter(torch.tensor(0.5))
        self.drug_weight = nn.Parameter(torch.tensor(0.5))


    def forward(self, cell_graph, drug_vector, pathway_tensor=None):

        cell_embedding = self.gnn(
            x=cell_graph.x,
            edge_index=cell_graph.edge_index,
            edge_attr=cell_graph.edge_attr if 'edge_attr' in cell_graph else None,
            pathway_tensor=pathway_tensor if pathway_tensor is not None else None,
            batch=cell_graph.batch 
        )
        
        if cell_embedding.dim() == 3 and cell_embedding.size(2) == 1: 
            cell_embedding = cell_embedding.squeeze(-1) 
        

        drug_embedding = self.drug_mlp(drug_vector.float())

        assert cell_embedding.shape == drug_embedding.shape, f"Shape mismatch: {cell_embedding.shape} vs {drug_embedding.shape}"

        combined_embedding = (cell_embedding * torch.sigmoid(self.cell_weight)) + \
                             (drug_embedding * torch.sigmoid(self.drug_weight))
        
        return self.resnet(combined_embedding)

class DrugMLP(nn.Module):
    def __init__(self, input_dim, embed_dim=44):
        
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
       
        return self.model(x)


