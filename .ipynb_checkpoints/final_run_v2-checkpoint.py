import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
from scripts import *
import scripts
import numpy as np
import pandas as pd 
from torch import nn
from torch.utils.data import Subset 
from torchmetrics import MetricTracker
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def ddp_setup(rank, world_size):
    """Setup the distributed environment for DDP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
def setup_config(pathway_groups):
    """Return the configuration dictionary"""
    config = {
        "gnn": {
            "input_dim": 1,
            "hidden_dim": 128,
            "output_dim": 1,
            "pathway_groups": pathway_groups, 
            "layer_modes": [True, False, True],
            "pooling_mode": "pathway",
            "aggr_modes": ["mean", "max", "mean"],
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
            "learning_rate": 1e-5,
            "batch_size": 4,  
            "clip_norm": 1.0,
            "stopping_patience": 10,
        },
        "env": {
            "max_epochs": 50,
            "world_size": torch.cuda.device_count(),  
            "local_rank": 0,  
            "master_addr": "localhost",  
            "master_port": "12356",  
            "save_every": 10
        }
    }
    return config



def load_train_objs():
    train_set, val_set, test_set, pathway_groups = scripts.get_data()
    config = setup_config(pathway_groups)
    gnn = scripts.ModularGNN(
          input_dim=config["gnn"]["input_dim"],
          hidden_dim=config["gnn"]["hidden_dim"],
          output_dim=config["gnn"]["output_dim"],
          pathway_groups=config["gnn"]["pathway_groups"],
          layer_modes=config["gnn"]["layer_modes"],
          aggr_modes=config["gnn"]["aggr_modes"],
          pooling_mode=config["gnn"]["pooling_mode"],
          num_pathways_per_instance=config["gnn"]["num_pathways_per_instance"])
    drug_mlp=scripts.DrugMLP(
          input_dim=config["drug"]["input_dim"],
          embed_dim=config["drug"]["embed_dim"])
    resnet=scripts.ResNet(
          embed_dim=config["resnet"]["embed_dim"],
          hidden_dim=config["resnet"]["hidden_dim"]
    )
    model = scripts.CombinedModel(gnn,drug_mlp,resnet)  # Assuming CombinedModel is defined in scripts
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['learning_rate'])
    return train_set,val_set,test_set,pathway_groups, model, optimizer, config


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )



def train_step(model, optimizer, train_data, config, device):
    """
    Perform a single training step for the model.
    Only computes MSE loss during training.
    """
    loss_fn = nn.MSELoss()
    total_loss = 0
    model.train()

    for batch_idx, batch in enumerate(train_data, start=1):
        try:
            # Unpack the batch
            cell_graph_batch, drug_tensor_batch, target_batch, _, _ = batch

            # Move data to the device
            cell_graph = cell_graph_batch.to(device)
            drug_vector = drug_tensor_batch.to(device)
            targets = target_batch.to(device)

            optimizer.zero_grad()

            # Forward pass through the CombinedModel
            outputs = model(cell_graph, drug_vector)
            outputs = outputs.view(-1)
            targets = targets.view(-1)

            # Compute loss
            loss = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
            optimizer.step()

            total_loss += loss.item()
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    return total_loss / len(train_data)






def train_model(model, optimizer, train_data, val_data, epoch, rank, config):
    
    device = torch.device(f'cuda:{rank}')
    early_stop = scripts.EarlyStop(config["optimizer"]["stopping_patience"])

    for epoch in range(config["env"]["max_epochs"]):
        # Training step
        train_loss = train_step(model, optimizer, train_data, config, device)

        # Log training loss
        if rank == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")

        # Validation step
        if val_data is not None:
            validation_metrics = scripts.evaluate_step(model, val_data, metrics, device)
            if rank == 0:
                print(f"Validation Metrics: {validation_metrics}")

        # Early stopping
        if rank == 0 and early_stop(train_loss):
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    return model

class Trainer:
    def __init__(self, model, train_data, val_data, test_data, optimizer, rank, save_every, config):
        self.rank = rank
        self.model = model.to(rank)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.config = config
        self.model = DDP(model, device_ids=[rank])

        # Initialize the metric tracker with GroupwiseMetric
        self.metrics = MetricTracker(torchmetrics.MetricCollection(
            {"R_cellwise_residuals": scripts.GroupwiseMetric(
                metric=torchmetrics.functional.pearson_corrcoef,
                grouping="drugs",
                average="macro",
                residualize=True),
             "R_cellwise": scripts.GroupwiseMetric(
                metric=torchmetrics.functional.pearson_corrcoef,
                grouping="cell_lines",
                average="macro",
                residualize=False),
             "MSE": torchmetrics.MeanSquaredError()  
            }))

        self.metrics.to(rank)
        self.early_stop = scripts.EarlyStop(config["optimizer"]["stopping_patience"])

    def train(self):
        for epoch in range(self.config["env"]["max_epochs"]):
            # Train the model
            train_loss = train_step(self.model, self.optimizer, self.train_data, self.config, self.rank)
            if self.rank == 0:
                print(f"Epoch {epoch}: Training Loss = {train_loss:.4f}")
                
            print("before eval")
            # Validate the model
            validation_metrics = self.evaluate_metrics(self.val_data)
            if self.rank == 0:
                print(f"Validation Metrics: {validation_metrics}")

            # Save periodically
            if epoch % self.save_every == 0 and self.rank == 0:
                print(f"[INFO] Checkpoint at Epoch {epoch}")

            # Early stopping check
            if self.early_stop(train_loss):
                print(f"Early stopping at epoch {epoch}.")
                break
    def evaluate_metrics(self, data_loader):
        """Evaluate metrics on the validation or test data."""
        print("in evaluate metrics")
        device = torch.device(f'cuda:{self.rank}')
        return scripts.evaluate_step(self.model, data_loader, self.metrics, device)

    def test(self):
        """Evaluate the model on the test data."""
        test_metrics = self.evaluate_metrics(self.test_data)
        if self.rank == 0:
            print(f"Test Metrics: {test_metrics}")

    def early_stop(self, train_loss):
        """Implement early stopping logic."""
        return self.early_stop(train_loss)

    def cleanup(self):
        dist.destroy_process_group()



def get_subset(dataset, fraction=0.2, seed=420):
    torch.manual_seed(seed) 
    num_samples = int(len(dataset) * fraction)
    indices = torch.randperm(len(dataset))[:num_samples] 
    return Subset(dataset, indices)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main(rank: int, world_size: int):

    try:
        # Step 1: Set up the process group for distributed training
        ddp_setup(rank, world_size)
        
        # Step 2: Load training objects (datasets, model, optimizer, etc.)
        train_set, val_set, test_set, pathway_groups, model, optimizer, config = load_train_objs()
        
        #### Subset datasets for faster training and debugging ####
        train_subset = get_subset(train_set, fraction=0.001)
        val_subset = get_subset(val_set, fraction=0.001)
        test_subset = get_subset(test_set, fraction=0.001)
        ###########################################################
        
        # Step 3: Prepare the distributed data loaders
        train_data = prepare_dataloader(train_subset, batch_size=config["optimizer"]["batch_size"])
        val_data = prepare_dataloader(val_subset, batch_size=config["optimizer"]["batch_size"])
        test_data = prepare_dataloader(test_subset, batch_size=config["optimizer"]["batch_size"])

        # Step 4: Initialize the trainer object
        trainer = Trainer(model, train_data, val_data, test_data, optimizer, rank,config["env"]["save_every"],config)
        
        # Step 5: Start the training and testing process
        trainer.train()
        trainer.test()
    
    except Exception as e:
        print(f"[ERROR] Rank {rank} encountered an error: {e}", flush=True)
    
    finally:
        # Step 6: Ensure the NCCL process group is properly destroyed
        if dist.is_initialized():  # Check if the process group is still active
            dist.destroy_process_group()


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    world_size = torch.cuda.device_count()
    print(f"world size:{world_size}")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)