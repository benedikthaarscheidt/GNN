#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append(r"scripts")
from functools import lru_cache
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import scripts
from scripts import *
import torchmetrics
from torch import nn
from torch_geometric.loader import DataLoader

def setup_environment(local_rank, world_size):
    """Setup the distributed environment for DDP."""
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)



@lru_cache(maxsize=None)
def get_data(n_fold=0, fp_radius=2):
    """Download, process, and prepare data for use in graph-based machine learning models."""
    import os
    import zipfile
    import requests
    import torch
    import pandas as pd
    import numpy as np
    import networkx as nx
    from torch_geometric.data import Data
    import scripts  # Assuming scripts has required functions

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



def setup_config(pathway_groups):
    """Return the configuration dictionary"""
    config = {
        "gnn": {
            "input_dim": 1,
            "hidden_dim": 128,
            "output_dim": 1,
            "pathway_groups": pathway_groups, 
            "layer_modes": [True, True, True],
            "pooling_mode": "pathway",
            "aggr_modes": ["mean", "mean", "mean"],
            "num_pathways_per_instance": 44
        },
        "resnet": {
            "embed_dim": 44,
            "hidden_dim": 128,
            "n_layers": 6,
            "dropout": 0.1,
        },
        "drug": {
            "input_dim": 2048,
            "embed_dim": 44
        },
        "optimizer": {
            "learning_rate": 1e-3,
            "batch_size": 8,  
            "clip_norm": 1.0,
            "stopping_patience": 10,
        },
        "env": {
            "max_epochs": 50,
            "world_size": torch.cuda.device_count(),  
            "local_rank": 0,  
            "master_addr": "localhost",  
            "master_port": "12345",  
        }
    }
    return config



def train_process(local_rank, train_data, val_data, pathway_groups, config):
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if local_rank == 0:
        print("[INFO] Data loaded successfully. Starting DDP training...")
    
    config["env"]["local_rank"] = local_rank
    config["env"]["rank"] = local_rank
    
    model = scripts.CombinedModel(config).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    R, model = scripts.train_model(config, train_data, val_data)
    
    dist.destroy_process_group()


def evaluate_step_ddp(model, loader, metrics, device, local_rank):
    metrics.reset()
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            cell_graph, drug_vector, targets = batch
            cell_graph = cell_graph.to(device)
            drug_vector = drug_vector.to(device)
            targets = targets.to(device)
            
            outputs = model(cell_graph, drug_vector)
            metrics.update(outputs.squeeze(), targets.squeeze())
    
    local_metrics = metrics.compute()
    aggregated_metrics = {}
    
    for key, value in local_metrics.items():
        tensor = value.clone().detach()
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            tensor /= dist.get_world_size()
        aggregated_metrics[key] = tensor.item()
    
    return aggregated_metrics


def save_datasets_to_file(train_data, val_data, test_data, pathway_groups, config, path="./"):
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'pathway_groups': pathway_groups,
        'config': config
    }, os.path.join(path, "datasets.pth"))
    print(f"[INFO] Saved datasets to {os.path.join(path, 'datasets.pth')}")


def load_datasets_from_file(path="./"):
    data_path = os.path.join(path, "datasets.pth")
    print(f"[INFO] Loading datasets from {data_path}")
    datasets = torch.load(data_path)
    return datasets['train_data'], datasets['val_data'], datasets['test_data'], datasets['pathway_groups'], datasets['config']


def main():
    local_rank, world_size = setup_environment()
    
    if local_rank == 0:
        print("[INFO] Rank 0 loading data...")
        train_data, val_data, test_data, pathway_groups = get_data()
        config = setup_config(pathway_groups)
        save_datasets_to_file(train_data, val_data, test_data, pathway_groups, config, path="./")
        print("[INFO] Rank 0 saved datasets to disk.")
    
    dist.barrier()
    
    train_data, val_data, test_data, pathway_groups, config = load_datasets_from_file(path="./")
    print(f"[INFO] Rank {local_rank} loaded datasets successfully.")
    
    train_process(local_rank, train_data, val_data, pathway_groups, config)


if __name__ == "__main__":
    try:
        local_rank, world_size = setup_environment()
        main()
    except Exception as e:
        print(f"[ERROR] Rank {local_rank} encountered an error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
