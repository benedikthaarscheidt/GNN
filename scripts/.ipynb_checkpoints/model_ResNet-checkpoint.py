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
        self.add_state("target", default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state("pred", default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state("drugs", default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state("cell_lines", default=torch.tensor([]), dist_reduce_fx='cat')

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

    def update(self, preds: Tensor, target: Tensor, cell_lines: Tensor, drugs: Tensor):
        """Update the metric states with new predictions and targets."""
        self.target = torch.cat([self.target, target], dim=0)
        self.pred = torch.cat([self.pred, preds], dim=0)
        self.drugs = torch.cat([self.drugs, drugs], dim=0)
        self.cell_lines = torch.cat([self.cell_lines, cell_lines], dim=0)

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
                    nn.Linear(hidden_dim, embed_dim)  # Make sure this goes back to embed_dim
                )
            )
        
        self.final_layer = nn.Linear(embed_dim, 1)  # Final prediction layer (optional)
        self._init_weights()  # Initialize weights using xavier uniform

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # Bias set to 0

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (layer(x) + x) * 0.5  # Residual connection with stabilization
        x = self.final_layer(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, gnn, drug_mlp, resnet):

        super().__init__()
        self.gnn = gnn
        self.drug_mlp = drug_mlp
        self.resnet = resnet

    def forward(self, cell_graph, drug_vector, pathway_tensor=None):

        # Cell embedding via GNN
        cell_embedding = self.gnn(
            x=cell_graph.x,
            edge_index=cell_graph.edge_index,
            edge_attr=cell_graph.edge_attr if 'edge_attr' in cell_graph else None,
            pathway_tensor=pathway_tensor if pathway_tensor is not None else None,
            batch=cell_graph.batch 
        )
        
        # Handle extra dimensions in the GNN output
        if cell_embedding.dim() == 3 and cell_embedding.size(2) == 1: 
            cell_embedding = cell_embedding.squeeze(-1)  # Squeeze only the third dimension
        
        # Drug embedding via DrugMLP
        drug_embedding = self.drug_mlp(drug_vector.float())

        # Shape consistency check
        assert cell_embedding.shape == drug_embedding.shape, f"Shape mismatch: {cell_embedding.shape} vs {drug_embedding.shape}"

        # Combine embeddings
        combined_embedding = (cell_embedding + drug_embedding) * 0.5  # Stabilized combination
        
        # Pass the combined embedding to ResNet
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
