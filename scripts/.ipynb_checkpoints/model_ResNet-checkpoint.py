import torch
from torch import nn
from torch import Tensor
from torchmetrics import Metric
import torchmetrics
from model_GNN import ModularGNN  


class EarlyStop():
    def __init__(self, max_patience, maximize=False):
        self.maximize = maximize
        self.max_patience = max_patience
        self.best_loss = None
        self.patience = max_patience + 0

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        elif loss < self.best_loss:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        else:
            self.patience -= 1
        return not bool(self.patience)


class GroupwiseMetric(Metric):
    def __init__(self, metric, grouping="cell_lines", average="macro", nan_ignore=False,
                 alpha=0.00001, residualize=False, **kwargs):
        super().__init__(**kwargs)
        self.grouping = grouping
        self.metric = metric
        self.average = average
        self.nan_ignore = nan_ignore
        self.residualize = residualize
        self.alpha = alpha
        self.add_state("target", default=torch.tensor([]))
        self.add_state("pred", default=torch.tensor([]))
        self.add_state("drugs", default=torch.tensor([]))
        self.add_state("cell_lines", default=torch.tensor([]))

    def get_residual(self, X, y):
        w = self.get_linear_weights(X, y)
        r = y - (X @ w)
        return r

    def get_linear_weights(self, X, y):
        A = X.T @ X
        Xy = X.T @ y
        n_features = X.size(1)
        A.flatten()[:: n_features + 1] += self.alpha
        return torch.linalg.solve(A, Xy).T

    def compute(self) -> Tensor:
        if self.grouping == "cell_lines":
            grouping = self.cell_lines
        elif self.grouping == "drugs":
            grouping = self.drugs
        metric = self.metric
        y_obs = self.target
        y_pred = self.pred
        unique = grouping.unique()
        metrics = [metric(y_obs[grouping == g], y_pred[grouping == g]) for g in unique]
        return torch.mean(torch.stack(metrics))  # Macro average


class CombinedModel(nn.Module):
    def __init__(self, gnn, drug_mlp, resnet):
        super().__init__()
        self.gnn = gnn
        self.drug_mlp = drug_mlp
        self.resnet = resnet

    def forward(self, cell_graph, drug_vector, pathway_tensor=None):

        print("Start of forward pass")  
        cell_embedding = self.gnn(
            x=cell_graph.x,
            edge_index=cell_graph.edge_index,
            edge_attr=cell_graph.edge_attr if 'edge_attr' in cell_graph else None,
            pathway_tensor=pathway_tensor if pathway_tensor is not None else None  
        )
        print("After GNN embedding") 
        print(f"cell embedding shape",cell_embedding.shape)

        if cell_embedding.dim() > 2: 
            
            cell_embedding = cell_embedding.mean(dim=0)  # Global mean pooling
            print("pooled bc dimensions of the graph embedding was off")

        drug_embedding = self.drug_mlp(drug_vector.float())
        print(f"Drug embedding shape: {drug_embedding.shape}")  

        combined_embedding = cell_embedding.float() + drug_embedding.float()

        return self.resnet(combined_embedding.float())

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

class ResNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim)
            ))
        self.final_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return self.final_layer(x)


#def train_step(model, optimizer, loader, config, device):
#    loss_fn = nn.MSELoss()
#    ls = []
#    model.train()
#    for c, d, targets, cell_lines, drugs in loader:
#        optimizer.zero_grad()
#        c, d, targets = c.to(device), d.to(device), targets.to(device)
#        outputs = model(c, d)
#        loss = loss_fn(outputs.squeeze(), targets.squeeze())
#        loss.backward()
#        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
#        optimizer.step()
#        ls.append(loss.item())
#    return np.mean(ls)
#
#
#def evaluate_step(model, loader, metrics, device):
#    metrics.increment()
#    model.eval()
#    for c, d, targets, cell_lines, drugs in loader:
#        with torch.no_grad():
#            c, d = c.to(device), d.to(device)
#            outputs = model(c, d)
#            metrics.update(
#                outputs.squeeze(),
#                targets.to(device).squeeze(),
#                cell_lines=cell_lines.to(device).squeeze(),
#                drugs=drugs.to(device).squeeze(),
#            )
#    return {key: value.item() for key, value in metrics.compute().items()}
