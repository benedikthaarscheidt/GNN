from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
import torch
import sys
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
import os
import networkx as nx
import scripts
from scripts import *
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import Metric
import pandas as pd 


def get_data(n_fold=0, fp_radius=2):
    def download_if_not_present(url, filepath):
        """Download a file from a URL if it does not exist locally."""
        if not os.path.exists(filepath):
            print(f"File not found at {filepath}. Downloading...")
            response = requests.get(url, stream=True)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("Download completed.")
        else:
            print(f"File already exists at {filepath}.")

    # Step 1: Download and load RNA-seq data
    zip_url = "https://cog.sanger.ac.uk/cmp/download/rnaseq_all_20220624.zip"
    zip_filepath = "data/rnaseq.zip"
    rnaseq_filepath = "data/rnaseq_normcount.csv"
    if not os.path.exists(rnaseq_filepath):
        download_if_not_present(zip_url, zip_filepath)
        with zipfile.ZipFile(zip_filepath, "r") as zipf:
            zipf.extractall("data/")
    rnaseq = pd.read_csv(rnaseq_filepath, index_col=0)

    # Step 2: Load gene network, hierarchies, and driver genes
    hierarchies = pd.read_csv("data/gene_to_pathway_final_with_hierarchy.csv")
    driver_genes = pd.read_csv("data/driver_genes_2.csv")['gene'].dropna()
    gene_network = nx.read_edgelist("data/filtered_gene_network.edgelist", nodetype=str)
    ensembl_to_hgnc = dict(zip(hierarchies['Ensembl_ID'], hierarchies['HGNC']))
    mapped_gene_network = nx.relabel_nodes(gene_network, ensembl_to_hgnc)

    # Step 3: Filter RNA-seq data and identify valid nodes
    driver_columns = rnaseq.columns.isin(driver_genes)
    filtered_rna = rnaseq.loc[:, driver_columns]
    valid_nodes = set(filtered_rna.columns)  # Get valid nodes after filtering RNA-seq columns

    # Step 4: Create edge tensors for the graph
    edges_df = pd.DataFrame(
        list(mapped_gene_network.edges(data="weight")),
        columns=["source", "target", "weight"]
    )
    edges_df["weight"] = edges_df["weight"].fillna(1.0).astype(float)
    filtered_edges = edges_df[
        (edges_df["source"].isin(valid_nodes)) & (edges_df["target"].isin(valid_nodes))
    ]
    node_to_idx = {node: idx for idx, node in enumerate(valid_nodes)}
    filtered_edges["source_idx"] = filtered_edges["source"].map(node_to_idx)
    filtered_edges["target_idx"] = filtered_edges["target"].map(node_to_idx)
    edge_index = torch.tensor(filtered_edges[["source_idx", "target_idx"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(filtered_edges["weight"].values, dtype=torch.float32)

    # Step 5: Process the hierarchy to create pathway groups
    filtered_hierarchy = hierarchies[hierarchies["HGNC"].isin(valid_nodes)]
    pathway_dict = {
        gene: pathway.split(':', 1)[1].split('[', 1)[0].strip() if isinstance(pathway, str) and ':' in pathway else None
        for gene, pathway in zip(filtered_hierarchy['HGNC'], filtered_hierarchy['Level_1'])
    }
    grouped_pathway_dict = {}
    for gene, pathway in pathway_dict.items():
        if pathway:
            grouped_pathway_dict.setdefault(pathway, []).append(gene)
    pathway_groups = {
        pathway: [node_to_idx[gene] for gene in genes if gene in node_to_idx]
        for pathway, genes in grouped_pathway_dict.items()
    }
    # Convert to padded tensor
    pathway_tensors = pad_sequence(
        [torch.tensor(indices, dtype=torch.long) for indices in pathway_groups.values()], 
        batch_first=True, 
        padding_value=-1  # Use -1 as padding
    )

    # Step 6: Create cell-line graphs
    tensor_exp = torch.tensor(filtered_rna.to_numpy())
    cell_dict = {cell: tensor_exp[i] for i, cell in enumerate(filtered_rna.index.to_numpy())}
    graph_data_list = {}
    for cell, x in cell_dict.items():
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.T
        elif x.ndim == 1:
            x = x.unsqueeze(1)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_data.y = None
        graph_data.cell_line = cell
        graph_data_list[cell] = graph_data

    # Step 7: Load drug data
    smile_dict = pd.read_csv("data/smiles.csv", index_col=0)
    fp = scripts.FingerprintFeaturizer(R=fp_radius)
    drug_dict = fp(smile_dict.iloc[:, 1], smile_dict.iloc[:, 0])

    # Step 8: Load IC50 data and filter for valid cell lines and drugs
    data = pd.read_csv("data/GDSC1.csv", index_col=0)
    data = data.query("SANGER_MODEL_ID in @cell_dict.keys() & DRUG_ID in @drug_dict.keys()")

    # Step 9: Split the data into folds for cross-validation
    unique_cell_lines = data["SANGER_MODEL_ID"].unique()

    np.random.seed(420)
    np.random.shuffle(unique_cell_lines)
    folds = np.array_split(unique_cell_lines, 10)
    train_idxs = list(range(10))
    train_idxs.remove(n_fold)
    validation_idx = np.random.choice(train_idxs)
    train_idxs.remove(validation_idx)
    train_lines = np.concatenate([folds[idx] for idx in train_idxs])
    validation_lines = folds[validation_idx]
    test_lines = folds[n_fold]

    train_data = data.query("SANGER_MODEL_ID in @train_lines")

    validation_data = data.query("SANGER_MODEL_ID in @validation_lines")
    test_data = data.query("SANGER_MODEL_ID in @test_lines")

    # Step 10: Build the datasets for training, validation, and testing
    train_dataset = scripts.OmicsDataset(graph_data_list, drug_dict, train_data)
    validation_dataset = scripts.OmicsDataset(graph_data_list, drug_dict, validation_data)
    test_dataset = scripts.OmicsDataset(graph_data_list, drug_dict, test_data)

    return train_dataset, validation_dataset, test_dataset, pathway_tensors


def custom_collate_fn(batch):
    try:
        cell_graphs = [item[0] for item in batch if item[0] is not None]  
        drug_vectors = torch.stack([item[1] for item in batch if item[1] is not None])  
        targets = torch.stack([item[2] for item in batch if item[2] is not None])  
        cell_ids = torch.stack([item[3] for item in batch if item[3] is not None]) 
        drug_ids = torch.stack([item[4] for item in batch if item[4] is not None]) 

        # Ensure cell_ids and drug_ids are long tensors (integers)
        cell_ids = cell_ids.long()
        drug_ids = drug_ids.long()

        cell_graph_batch = Batch.from_data_list(cell_graphs)  
        return cell_graph_batch, drug_vectors, targets, cell_ids, drug_ids
    
    except Exception as e:
        print(f"Error in custom_collate_fn: {e}")
        print(f"Batch contents: {batch}")
        raise e


def evaluate_step(model, loader, metrics, device):
    print("[INFO] Starting evaluation step")
    metrics.increment()
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            try:
                # Unpack the batch
                cell_graph_batch, drug_tensor_batch, target_batch, cell_id_batch, drug_id_batch = batch

                # Move data to the device
                cell_graph = cell_graph_batch.to(device)
                drug_vector = drug_tensor_batch.to(device)
                targets = target_batch.to(device)
                cell_ids = cell_id_batch.to(device).long()
                drug_ids = drug_id_batch.to(device).long()

                # Forward pass through the CombinedModel
                outputs = model(cell_graph, drug_vector)

                # Debugging outputs and shapes
                print(f"[DEBUG] Raw Outputs Shape: {outputs.shape}")

                # Ensure outputs have consistent dimensions
                outputs = outputs.view(-1)  # Flatten into 1D tensor for batch processing
                targets = targets.view(-1)
                cell_ids = cell_ids.view(-1)
                drug_ids = drug_ids.view(-1)

                # Guard against empty or invalid tensors
                if outputs.numel() == 0 or targets.numel() == 0 or cell_ids.numel() == 0 or drug_ids.numel() == 0:
                    print("[WARNING] Empty or invalid tensors encountered. Skipping batch.")
                    continue

                print(f"[DEBUG] Final Outputs Shape: {outputs.shape}")
                print(f"[DEBUG] Targets Shape: {targets.shape}")
                print(f"[DEBUG] Cell IDs Shape: {cell_ids.shape}, Drug IDs Shape: {drug_ids.shape}")

                # Update metrics
                metrics.update(outputs, targets, cell_lines=cell_ids, drugs=drug_ids)

            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} encountered an error: {e}")
                continue

    # Compute final metrics
    return {key: value.item() for key, value in metrics.compute().items()}





class EarlyStop():
    def __init__(self, max_patience, maximize=False):
        self.maximize=maximize
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
    def __init__(self, metric,
                 grouping = "cell_lines",
                 average = "macro",
                 nan_ignore=False,
                 alpha=0.00001,
                 residualize = False,
                 **kwargs):
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
        r = y-(X@w)
        return r
    def get_linear_weights(self, X, y):
        A = X.T@X
        Xy = X.T@y
        n_features = X.size(1)
        A.flatten()[:: n_features + 1] += self.alpha
        return torch.linalg.solve(A, Xy).T
    def get_residual_ind(self, y, drug_id, cell_id, alpha=0.001):
        X = torch.cat([y.new_ones(y.size(0), 1),
                       torch.nn.functional.one_hot(drug_id),
                       torch.nn.functional.one_hot(cell_id)], 1).float()
        return self.get_residual(X, y)

    def compute(self) -> Tensor:
        if self.grouping == "cell_lines":
            grouping = self.cell_lines
        elif self.grouping == "drugs":
            grouping = self.drugs
        metric = self.metric
        if not self.residualize:
            y_obs = self.target
            y_pred = self.pred
        else:
            y_obs = self.get_residual_ind(self.target, self.drugs, self.cell_lines)
            y_pred = self.get_residual_ind(self.pred, self.drugs, self.cell_lines)
        average = self.average
        nan_ignore = self.nan_ignore
        unique = grouping.unique()
        proportions = []
        metrics = []
        for g in unique:
            is_group = grouping == g
            metrics += [metric(y_obs[grouping == g], y_pred[grouping == g])]
            proportions += [is_group.sum()/len(is_group)]
        if average is None:
            return torch.stack(metrics)
        if (average == "macro") & (nan_ignore):
            return torch.nanmean(y_pred.new_tensor([metrics]))
        if (average == "macro") & (not nan_ignore):
            return torch.mean(y_pred.new_tensor([metrics]))
        if (average == "micro") & (not nan_ignore):
            return (y_pred.new_tensor([proportions])*y_pred.new_tensor([metrics])).sum()
        else:
            raise NotImplementedError
    
    def update(self, preds: Tensor, target: Tensor,  drugs: Tensor,  cell_lines: Tensor) -> None:

        self.target = torch.cat([self.target, target])
        self.pred = torch.cat([self.pred, preds])
        self.drugs = torch.cat([self.drugs, drugs]).long()
        self.cell_lines = torch.cat([self.cell_lines, cell_lines]).long()

































