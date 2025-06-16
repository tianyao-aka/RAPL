import os
# Must be set *before* importing torch or allocating GPU tensors:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import sys
import pandas as pd
import time
import torch
import torch.nn.functional as F

# from src.model.Trainer import Trainer
from src.model.Trainerv3 import Trainer
from src.dataset.PostProcessedDataset import ProcessedDiskDataset
from src.dataset.utils import load_pickles,decode_path
from termcolor import colored
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def color_print(text, color='green'):
    print(colored(text, color))

def main():
    parser = argparse.ArgumentParser(description="Train a GNN-based model with path prediction.")

    # Add arguments for Trainer configuration
    parser.add_argument("--model_type", type=str, default="GCN", help="Type of GNN model: GCN, GAT, SGConv, GIN.")
    parser.add_argument("--bidirectional", action="store_true",default=False, help="Use bidirectional GNN if set.")
    parser.add_argument("--use_stop_mlp", action="store_true",default=False, help="Use a separate MLP for stop token if set.")
    parser.add_argument("--num_mlp_layers", type=int, default=2, help="Number of MLP layers.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in GAT.")
    parser.add_argument("--K", type=int, default=3, help="K for SGConv.")
    parser.add_argument("--in_dims", type=int, default=3072, help="Input node feature dimension.")
    parser.add_argument("--hidden_dims", type=int, default=512, help="Hidden dimension size in GNN.")
    parser.add_argument("--out_dims", type=int, default=512, help="Output dimension for GNN.")
    parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization if set.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--device", type=int, default=0, help="GPU device index. Use -1 for CPU.")
    
    # Add dataset-related arguments
    parser.add_argument("--dataset_name", type=str, default="webqsp", help="Name or path of the dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train, val, test.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the DataLoader.")
    
    parser.add_argument("--pathTrainAfterEpoch", type=int, default=0, help="start training path loss after #epoch")
    parser.add_argument("--wandb_id", type=str, default='1')

    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # -------------------------
    # 0. init wandb
    # -------------------------

    # Start a new wandb run to track this script.
    # run = wandb.init(
    #     # Set the wandb project where this run will be logged.
    #     project=f"KGQA_{args.dataset_name}_{args.model_type}_wandb_{args.wandb_id}",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "bidirectional": args.bidirectional,
    #         "num_layers": args.num_layers,
    #         "use_stop_mlp": args.use_stop_mlp,
    #         "epochs": 100,
    #         "num_heads": args.num_heads,
    #         "K": args.K
    #     },
    # )
    # args.wandb = run
    save_dir = f'experiments/{args.dataset_name}/saved_models/{args.wandb_id}/{args.model_type}_bidirectional_{args.bidirectional}_numLayers_{args.num_layers}_useStopMlp_{args.use_stop_mlp}_numHeads_{args.num_heads}_K_{args.K}/seed_{seed}/'
    os.makedirs(save_dir, exist_ok=True)
    
    
    color_print(args,'cyan')
    
    # -------------------------
    # 1. load datasets
    # -------------------------  
    dataset = args.dataset  
    train_set1 = ProcessedDiskDataset(processed_dir=f'data/post_processed_pathLabel/{dataset}/',split='train') 
    val_set1 = ProcessedDiskDataset(processed_dir=f'data/post_processed_pathLabel/{dataset}/',split='val')  

    
    # -------------------------
    # 2. Create DataLoader
    # -------------------------
    train_dataloader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True,num_workers=4)
    val_dataloader1 = DataLoader(val_set1, batch_size=args.batch_size, shuffle=False,num_workers=4)
    color_print(f"Created DataLoader with batch_size={args.batch_size}.",'red')
    
    # -------------------------
    # 3. Initialize Trainer
    # -------------------------
    # Map device: if args.device==-1, use CPU; else GPU

    device = args.device if torch.cuda.is_available() else "cpu"
    color_print (f"use_device:{device}",'red')
    args.device = device
    args = vars(args)
    trainer = Trainer(**args)

    color_print("Trainer initialized.",'red')

    # -------------------------
    # 4. Train the Model
    # -------------------------
    trainer.to(device)
    print ('formal training with 4o labels')
    trainer.fit_mixture(train_dataloader1,val_dataloader1,save_dir=save_dir,pathtrainingstart=True)

if __name__ == "__main__":
    main()

