import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchmetrics import Metric
import torchmetrics
from torch_geometric.nn import GCNConv, GINConv  # Example layers for GNN

# --- GNN Definitions ---
class ModularPathwayConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='sum', pathway_groups=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.pathway_groups = pathway_groups
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None, pathway_mode=False):
        x_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if not pathway_mode or self.pathway_groups is None:
            return x_updated
        x_updated = x_updated.clone()
        for pathway, nodes in self.pathway_groups.items():
            temp_x = x.clone()
            sub_edge_index, _ = torch_geometric.utils.subgraph(nodes, edge_index, relabel_nodes=False)
            valid_mask = torch.isin(sub_edge_index[0], nodes) & torch.isin(sub_edge_index[1], nodes)
            sub_edge_index = sub_edge_index[:, valid_mask]
            sub_edge_attr = edge_attr[valid_mask] if edge_attr is not None else None
            pathway_features = self.propagate(sub_edge_index, x=temp_x, edge_attr=sub_edge_attr)
            x_updated[nodes] = pathway_features[nodes]
        return x_updated

    def propagate(self, edge_index, x, edge_attr=None):
        row, col = edge_index
        messages = self.message(x[row], x[col], edge_attr)
        out = torch.zeros((x.size(0), messages.size(1)), device=x.device)
        out.index_add_(0, col, messages)
        return out

    def message(self, x_i, x_j, edge_attr=None):
        scaled_x_i = x_i * edge_attr.view(-1, 1) if edge_attr is not None else x_i
        combined_message = torch.cat([scaled_x_i, x_j], dim=1)
        return self.mlp(combined_message)


class ModularGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pathway_groups=None, layer_modes=None, aggr_modes=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.pathway_groups = pathway_groups
        layer_modes = layer_modes or [False] * 3
        aggr_modes = aggr_modes or ['sum'] * 3
        self.layers.append(ModularPathwayConv(input_dim, hidden_dim, aggr=aggr_modes[0], pathway_groups=pathway_groups))
        for mode, aggr in zip(layer_modes[1:-1], aggr_modes[1:-1]):
            self.layers.append(ModularPathwayConv(hidden_dim, hidden_dim, aggr=aggr, pathway_groups=pathway_groups))
        self.layers.append(ModularPathwayConv(hidden_dim, output_dim, aggr=aggr_modes[-1], pathway_groups=pathway_groups))

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


# --- Early Stopping ---
class EarlyStop:
    def __init__(self, max_patience, maximize=False):
        self.maximize = maximize
        self.max_patience = max_patience
        self.best_loss = None
        self.patience = max_patience

    def __call__(self, loss):
        if self.best_loss is None or (loss < self.best_loss) != self.maximize:
            self.best_loss = loss
            self.patience = self.max_patience
        else:
            self.patience -= 1
        return not bool(self.patience)


# --- ResNet ---
class ResNet(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1, n_layers=6, norm="layernorm"):
        super().__init__()
        self.mlps = nn.ModuleList()
        norm_layer = nn.LayerNorm if norm == "layernorm" else nn.BatchNorm1d if norm == "batchnorm" else nn.Identity
        for _ in range(n_layers):
            self.mlps.append(nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim)
            ))
        self.lin = nn.Linear(embed_dim, 1)

    def forward(self, x):
        for l in self.mlps:
            x = (l(x) + x) / 2
        return self.lin(x)


# --- Main Model ---
class Model(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1, n_layers=6, norm="layernorm", gnn=None):
        super().__init__()
        self.resnet = ResNet(embed_dim, hidden_dim, dropout, n_layers, norm)
        self.embed_d = nn.Sequential(nn.LazyLinear(embed_dim), nn.ReLU())
        self.gnn = gnn
        self.embed_c = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, c, d, graph_data):
        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
        edge_attr = graph_data.get('edge_attr', None)
        gnn_output = self.gnn(node_features, edge_index, edge_attr)
        c_embedding = gnn_output[c]
        c_embedding = self.embed_c(c_embedding)
        return self.resnet(self.embed_d(d) + c_embedding)


# --- Training & Evaluation ---
def train_step(model, optimizer, loader, config, device):
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = []
    for x in loader:
        optimizer.zero_grad()
        graph_data = {'node_features': x[5].to(device), 'edge_index': x[6].to(device)}
        if len(x) > 7:
            graph_data['edge_attr'] = x[7].to(device)
        out = model(x[0].to(device), x[1].to(device), graph_data)
        loss = loss_fn(out.squeeze(), x[2].to(device).squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
        optimizer.step()
        total_loss.append(loss.item())
    return np.mean(total_loss)


def evaluate_step(model, loader, metrics, device):
    metrics.increment()
    model.eval()
    for x in loader:
        with torch.no_grad():
            graph_data = {'node_features': x[5].to(device), 'edge_index': x[6].to(device)}
            if len(x) > 7:
                graph_data['edge_attr'] = x[7].to(device)
            out = model(x[0].to(device), x[1].to(device), graph_data)
            metrics.update(out.squeeze(),
                           x[2].to(device).squeeze(),
                           cell_lines=x[3].to(device).squeeze(),
                           drugs=x[4].to(device).squeeze())
    return {k: v.item() for k, v in metrics.compute().items()}


def train_model(config, train_dataset, validation_dataset=None):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["optimizer"]["batch_size"], shuffle=True)
    val_loader = None
    if validation_dataset:
        val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config["optimizer"]["batch_size"])
    model = Model(**config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning_rate"])
    device = torch.device(config["env"]["device"])
    model.to(device)
    early_stop = EarlyStop(config["optimizer"]["stopping_patience"])
    metrics = torchmetrics.MetricCollection({
        "MSE": torchmetrics.MeanSquaredError()
    }).to(device)
    for epoch in range(config["env"]["max_epochs"]):
        train_loss = train_step(model, optimizer, train_loader, config, device)
        print(f"Epoch {epoch}: Training Loss: {train_loss}")
        if val_loader:
            validation_metrics = evaluate_step(model, val_loader, metrics, device)
            print(f"Validation Metrics: {validation_metrics}")
        if early_stop(train_loss):
            break
    return model
