import sys
import os
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
from tqdm import tqdm
import numpy as np
import pandas as pd 
from torch import nn
from torch.utils.data import Subset 
from torchmetrics import Metric
import torchmetrics
import warnings
from multiprocessing import Process
from torch.cuda.amp import GradScaler, autocast
import GPUtil

warnings.filterwarnings("ignore", category=FutureWarning)


def adjust_accumulation_steps(accumulation_steps, memory_threshold=80):
    gpu = GPUtil.getGPUs()[0]  # Assuming single GPU
    memory_utilization = gpu.memoryUtil * 100  # Convert to percentage

    if memory_utilization > memory_threshold and accumulation_steps > 1:
        return accumulation_steps - 1  # Reduce steps
    elif memory_utilization < memory_threshold - 20:
        return accumulation_steps + 1  # Increase steps
    return accumulation_steps


def ddp_setup(rank, world_size):
    """Setup the distributed environment for DDP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    if torch.cuda.device_count() < world_size:
        raise ValueError(f"Insufficient GPUs: {torch.cuda.device_count()} available, but {world_size} required.")

    device = f"cuda:{rank}"  # Convert rank to device
    torch.cuda.set_device(device)  # Set the current device
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
def setup_config(pathway_groups):
    """Return the configuration dictionary"""
    config = {
        "gnn": {
            "input_dim": 1,
            "hidden_dim": 128,
            "output_dim": 1,
            "pathway_groups": pathway_groups, 
            "layer_modes": [True, False, True, False, True, False],
            "pooling_mode": "pathway",
            "aggr_modes": ["mean", "mean", "mean", "mean", "mean","mean"],
            "num_pathways_per_instance": 44
        },
        "resnet": {
            "embed_dim": 2816,
            "hidden_dim": 128,
            "n_layers": 6,
            "dropout": 0.1,
        },
        "drug": {
            "input_dim": 2048,
            "embed_dim": 2816,
        },
        "optimizer": {
            "learning_rate": 1e-4,
            "batch_size": 32,  
            "clip_norm": 1.0,
            "stopping_patience": 10,
            "accumulation_steps": 8,  # Initial accumulation steps
            "dynamic_adjust": True,  # Flag for dynamic adjustment
            "memory_threshold": 90,  # Memory utilization threshold (%)
        },
        "scheduler": {
            "step_size": 10,  # Step size for the scheduler
            "gamma": 0.7,  # Multiplicative factor for learning rate decay
        },
        "env": {
            "max_epochs": 10,
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
        num_pathways_per_instance=config["gnn"]["num_pathways_per_instance"]
    )
    drug_mlp = scripts.DrugMLP(
        input_dim=config["drug"]["input_dim"],
        embed_dim=config["drug"]["embed_dim"]
    )
    resnet = scripts.ResNet(
        embed_dim=config["resnet"]["embed_dim"],
        hidden_dim=config["resnet"]["hidden_dim"]
    )
    gnn.summary()
    model = scripts.CombinedModel(gnn, drug_mlp, resnet)
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["learning_rate"])

    # Define StepLR scheduler using config
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config["scheduler"]["step_size"], 
        gamma=config["scheduler"]["gamma"]
    )

    return train_set, val_set, test_set, pathway_groups, model, optimizer, scheduler, config

def prepare_dataloader(dataset: Dataset, batch_size: int, rank=0, distributed=True, shuffle=True):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle) if shuffle else DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=(sampler is None and shuffle),  # Shuffle only if no sampler is used and shuffle=True
        sampler=sampler,
        drop_last=True
    )




def train_step(model, optimizer, train_data, config, device, rank=0):
    loss_fn = nn.MSELoss()
    total_loss = 0
    model.train()

    scaler = GradScaler()
    accumulation_steps = config.get("accumulation_steps", 1)

    prefetch_iterator = iter(train_data)
    next_batch = next(prefetch_iterator, None)

    if rank == 0:
        pbar = tqdm(total=len(train_data), desc="Training", unit="batch", leave=False)
    else:
        pbar = None

    optimizer.zero_grad()
    
    for batch_idx in range(1, len(train_data) + 1):
        try:

            accumulation_steps = adjust_accumulation_steps(accumulation_steps)
            
            current_batch = next_batch
            next_batch = next(prefetch_iterator, None)

            if current_batch is None:
                break

            cell_graph_batch, drug_tensor_batch, target_batch, *_ = current_batch
            cell_graph = cell_graph_batch.to(device, non_blocking=True)
            drug_vector = drug_tensor_batch.to(device, non_blocking=True)
            targets = target_batch.to(device, non_blocking=True)

            with autocast():
                try:
                    outputs = model(cell_graph, drug_vector)
                except Exception as e:
                    print("model is faulty")

                    raise e
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            try:
                if batch_idx % accumulation_steps == 0 or next_batch is None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["optimizer"]["clip_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
            except Exception as e:
                print("accumulation")
                raise e  
            try:
                total_loss += loss.item() * accumulation_steps
            except Exception as e:
                print(f"Ich knall dir eine")
                raise e  
            if rank == 0 and pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"Batch Loss": loss.item() * accumulation_steps})

        except Exception as e:
            print(f"trainstep")
            raise e  

    if rank == 0 and pbar is not None:
        pbar.close()

    return total_loss / len(train_data)



    
class Trainer:
    def __init__(self, model, train_data, val_data, test_data, optimizer, scheduler, rank, save_every, config):
        self.rank = rank
        self.model = model.to(f"cuda:{rank}")
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.config = config
        


        self.early_stop = scripts.EarlyStop(config["optimizer"]["stopping_patience"])

    def train(self):
        for epoch in range(self.config["env"]["max_epochs"]):
            if isinstance(self.train_data.sampler, DistributedSampler):
                self.train_data.sampler.set_epoch(epoch)
    

            train_loss = train_step(
                model=self.model,
                optimizer=self.optimizer,
                train_data=self.train_data,
                config=self.config,
                device=f"cuda:{self.rank}",
                rank=self.rank
            )
            print(f"Epoch {epoch} on rank {self.rank}: Training Loss = {train_loss:.4f}")
    

            if self.rank == 0:
                validation_loss = self.evaluate("validation", self.val_data)
                print(f"Epoch {epoch} on rank {self.rank}: Validation MSE = {validation_loss:.6f}")
    

            dist.barrier()
    

            if epoch % self.save_every == 0 and self.rank == 0:
                print(f"[INFO] Saving checkpoint at Epoch {epoch}")

                torch.save({
                    "model_state": self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "epoch": epoch,
                }, f"checkpoint_epoch_{epoch}.pt")
    

            if self.early_stop(train_loss):
                print(f"Early stopping at epoch {epoch}.")
                break

                
#
    def early_stop(self, train_loss):

        return self.early_stop(train_loss)

    def cleanup(self):
        dist.destroy_process_group()


    def evaluate(self, dataset_name: str, dataset):
        
        if self.rank != 0:
            return
    
    

        loader = DataLoader(
            dataset,
            batch_size=self.config["optimizer"]["batch_size"],
            shuffle=False,
            pin_memory=True,
        )
    

        mse_loss = nn.MSELoss()
    

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.eval()
    
        total_loss = 0.0
        num_batches = 0
    
        with torch.no_grad():
            for batch in loader:
                cell_graph_batch, drug_tensor_batch, target_batch, *_ = batch
                cell_graph = cell_graph_batch.to(f"cuda:{self.rank}", non_blocking=True)
                drug_vector = drug_tensor_batch.to(f"cuda:{self.rank}", non_blocking=True)
                targets = target_batch.to(f"cuda:{self.rank}", non_blocking=True)
    
                outputs = model(cell_graph, drug_vector)
                loss = mse_loss(outputs.view(-1), targets.view(-1))
    
                total_loss += loss.item()
                num_batches += 1
    

        average_loss = total_loss / num_batches

        return average_loss

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

def main_process(rank: int, world_size: int):
    try:

        ddp_setup(rank, world_size)


        train_set, val_set, test_set, pathway_groups, model, optimizer, scheduler, config = load_train_objs()


        train_subset = get_subset(train_set, fraction=0.99)
        val_subset = get_subset(val_set, fraction=0.99)
        test_subset = get_subset(test_set, fraction=0.99)

        model = model.to(f"cuda:{rank}")

        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        train_data = prepare_dataloader(
            train_subset,
            batch_size=config["optimizer"]["batch_size"],
            rank=rank,
            shuffle=True,
        )

        trainer = Trainer(
            model=model,
            train_data=train_data,
            val_data=val_subset,
            test_data=test_subset,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            save_every=config["env"]["save_every"],
            config=config,
        )

        trainer.train()

        if rank == 0:

                test_loss = trainer.evaluate("test", test_subset)
                print(f"[INFO] Final Test MSE = {test_loss:.6f}")
    
    except Exception as e:
        print(f"[ERROR] Rank {rank} encountered an error: {e}", flush=True)

    finally:

        if dist.is_initialized():
            dist.destroy_process_group()




if __name__ == "__main__":
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    mp.set_start_method("spawn", force=True)  # Force the use of the 'spawn' start method
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    world_size = torch.cuda.device_count()  

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()