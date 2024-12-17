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
import torchmetrics
from torch import nn
import optuna
import models
from optuna.integration import TensorBoardCallback
from model_GNN import ModularPathwayConv, ModularGNN
torch.set_printoptions(threshold=torch.inf)
from model_ResNet import CombinedModel, ResNet, DrugMLP  
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def custom_collate_fn(batch):
    
    try:
        cell_graphs = [item[0] for item in batch if item[0] is not None]  
        drug_vectors = torch.stack([item[1] for item in batch if item[1] is not None])  
        targets = torch.stack([item[2] for item in batch if item[2] is not None])  
        cell_ids = torch.stack([item[3] for item in batch if item[3] is not None]) 
        drug_ids = torch.stack([item[4] for item in batch if item[4] is not None]) 


        cell_graph_batch = Batch.from_data_list(cell_graphs)  
        return cell_graph_batch, drug_vectors, targets, cell_ids, drug_ids
    
    except Exception as e:
        print(f"Error in custom_collate_fn: {e}")
        print(f"Batch contents: {batch}")
        raise e


def evaluate_step(model, loader, metrics, device):

    metrics.increment()
    model.eval() 

    with torch.no_grad(): 
        for batch in loader:

            cell_graph, drug_vector, targets = batch  
            cell_graph = cell_graph.to(device)  # Batch object (PyG Data)
            drug_vector = drug_vector.to(device)  # Tensor
            targets = targets.to(device)  # Tensor


            outputs = model(cell_graph, drug_vector)

            metrics.update(
                outputs.squeeze(),
                targets.squeeze()
            )

    return {key: value.item() for key, value in metrics.compute().items()}

def train_step(model, optimizer, loader, config, device):
    print(f"Running on device: {device}")  # Verify the GPU being used for training
    loss_fn = nn.MSELoss()
    total_loss = 0
    model.train()
    i = 0
    for batch in loader:
        i += 1
        try:
            cell_graph_batch, drug_tensor_batch, target_batch, cell_id_batch, drug_id_batch = batch
        except Exception as e:
            print(f"Error unpacking batch: {e}")
            print(f"Batch contents: {batch}")
            continue

        cell_graph = cell_graph_batch.to(device)
        drug_vector = drug_tensor_batch.to(device)
        targets = target_batch.to(device)

        optimizer.zero_grad()

        outputs = model(cell_graph, drug_vector)

        loss = loss_fn(outputs.squeeze(), targets.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {i}, Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def init_ddp(config, model):

    local_rank = config["env"]["local_rank"]

    # Manually set the environment variables for DistributedDataParallel
    os.environ['MASTER_ADDR'] = 'localhost'  # Master node address
    os.environ['MASTER_PORT'] = '12345'     # Communication port
    os.environ['WORLD_SIZE'] = str(config["env"]["world_size"])  # Total number of processes (GPUs)
    os.environ['RANK'] = str(config["env"]["rank"])  # Global rank
    os.environ['LOCAL_RANK'] = str(local_rank)  # Local rank

    # Set the device for the current process (GPU)
    torch.cuda.set_device(local_rank)

    # Move model to the correct GPU for this process
    model = model.to(f'cuda:{local_rank}')

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')

    # Wrap the model in DistributedDataParallel (DDP)
    model = DDP(model, device_ids=[local_rank])

    return model
def train_model(config, train_dataset, validation_dataset=None, callback_epoch=None):
    """Main training function for the combined GNN, ResNet, and DrugMLP model."""
    # Force single GPU setup (disable multi-GPU)
    config["env"]["world_size"] = 1  # Use only one GPU
    config["env"]["rank"] = 0        # Global rank for the first GPU
    config["env"]["local_rank"] = 0   # Local rank for the first GPU (GPU 0)


    train_sampler = DistributedSampler(train_dataset, num_replicas=config["env"]["world_size"], rank=config["env"]["rank"])

    # Set up DataLoader with the sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["optimizer"]["batch_size"],
        shuffle=False,  # Don't shuffle, since DistributedSampler handles it
        drop_last=True,
        collate_fn=custom_collate_fn,
        num_workers=8,  # Parallelize data loading
        sampler=train_sampler  # Use the sampler for distributed loading
    )
    
    val_loader = None
    if validation_dataset is not None:
        # Set up DistributedSampler for validation dataset
        val_sampler = DistributedSampler(validation_dataset, num_replicas=config["env"]["world_size"], rank=config["env"]["rank"], shuffle=False)
    
        val_loader = DataLoader(
            validation_dataset,
            batch_size=config["optimizer"]["batch_size"],
            shuffle=False,  # Don't shuffle, since DistributedSampler handles it
            drop_last=True,
            collate_fn=custom_collate_fn,
            num_workers=8,  # Parallelize data loading
            sampler=val_sampler  # Use the sampler for distributed loading
        )
    device = torch.device(config["env"]["device"])

    # Initialize the Combined Model
    gnn_model = ModularGNN(**config["gnn"])
    drug_mlp = DrugMLP(input_dim=config["drug"]["input_dim"], embed_dim=config["drug"]["embed_dim"])
    resnet = ResNet(embed_dim=config["drug"]["embed_dim"], hidden_dim=config["resnet"]["hidden_dim"])
    combined_model = CombinedModel(gnn=gnn_model, drug_mlp=drug_mlp, resnet=resnet)
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"number of GPUs:{n_gpus}")
            combined_model = init_ddp(config, combined_model)  # Initialize DDP if multiple GPUs are available
    
    combined_model.to(device)

    optimizer = torch.optim.Adam(combined_model.parameters(), lr=config["optimizer"]["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stop = EarlyStop(config["optimizer"]["stopping_patience"])

    if not hasattr(combined_model, 'lazy_initialized') or not combined_model.lazy_initialized:
        test_instance = next(iter(train_loader))
        cell_graph_batch, drug_vector, targets, cell_ids, drug_ids = test_instance
        cell_graph_batch = cell_graph_batch.to(device)
        drug_vector = drug_vector.to(device)
        with torch.no_grad():
            combined_model(cell_graph_batch, drug_vector)
        combined_model.lazy_initialized = True  # Track initialization state
        print("Lazy layers initialized successfully with a real batch instance.")

    metrics = torchmetrics.MetricTracker(torchmetrics.MetricCollection({
        "R_cellwise_residuals": GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef, grouping="drugs", average="macro", residualize=True),
        "R_cellwise": GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef, grouping="cell_lines", average="macro", residualize=False),
        "MSE": torchmetrics.MeanSquaredError()
    }))
    metrics.to(device)

    best_val_target = None
    for epoch in range(config["env"]["max_epochs"]):
        train_loss = train_step(combined_model, optimizer, train_loader, config, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")

        if val_loader is not None:
            validation_metrics = evaluate_step(combined_model, val_loader, metrics, device)
            best_val_target = validation_metrics.get('R_cellwise_residuals', None)

        if callback_epoch is not None:
            callback_epoch(epoch, best_val_target)

        if early_stop(train_loss):
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    return best_val_target, combined_model