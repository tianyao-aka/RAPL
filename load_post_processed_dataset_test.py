import numpy as np
import os
import sys
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever_v2 import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample
from src.dataset.PyG_dataset_disk import *
from src.dataset.utils import *
import pickle
import argparse

def show_non_zero_topic_candidates(data):
    non_zero_indices = (data.topic_candidates != 0).nonzero(as_tuple=True)[0]
    return non_zero_indices.tolist()

def show_zero_indices_path_label(data):
    zero_indices = {}
    for col in range(data.path_label.size(1)):
        indices = (data.path_label[:, col] == 0).nonzero(as_tuple=True)[0].tolist()
        if indices:
            zero_indices[col] = indices
    return zero_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset name.')
    parser.add_argument('--name', type=str, default='webqsp', help='Name of the dataset')
    args = parser.parse_args()
    name = args.name
    print ('making dataset:',name)
    dset = KGDataset(root=f'data/{name}/',split='val')
    dset = KGDataset(root=f'data/{name}/',split='test')
    print (dset[0])

    
    
    