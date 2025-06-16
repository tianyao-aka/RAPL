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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever_v2 import RetrieverDataset, collate_retriever
# from src.dataset.calc_path_distance import RetrieverDataset
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample
import pickle


def get_sorted_paths(val, top_k=10):
    """
    Sorts paths in ascending order of distance, truncates to top_k,
    and returns sorted/truncated translated_paths, reasoning_paths, and distances.

    Parameters
    ----------
    val : dict
        Must contain:
            val['translated_paths'] : list of str
            val['reasoning_paths'] : list of list/tuple
            val['path_distances']  : list of int
    top_k : int
        The maximum number of paths to return.

    Returns
    -------
    sorted_translated_paths : list of str
        The translated_paths sorted by ascending distance, truncated to top_k.
    sorted_reasoning_paths : list of list/tuple
        The reasoning_paths sorted by ascending distance, truncated to top_k.
    sorted_distances : list of int
        The corresponding distances sorted in ascending order, truncated to top_k.
    """
    # Convert distances to numpy array for argsort
    distances = np.array(val['path_distances'])

    # Get sorted indices based on distance (ascending order)
    sorted_indices = np.argsort(distances)

    # Retrieve the sorted and truncated paths
    sorted_translated_paths = [val['translated_paths'][i] for i in sorted_indices[:top_k]]
    sorted_reasoning_paths = [val['reasoning_paths'][i] for i in sorted_indices[:top_k]]
    sorted_distances = [val['path_distances'][i] for i in sorted_indices[:top_k]]

    return sorted_translated_paths, sorted_reasoning_paths, sorted_distances


def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    exp_name = f'{exp_prefix}_{ts}'

    os.makedirs(exp_name, exist_ok=True)
    train_set = RetrieverDataset(config=config, split='train')

    
    N = len(train_set)
    train_dict_list = []
    val_dict_list = []
    test_dict_list = []
    # # train data
    print ('train_data processing')
    for i in tqdm(range(N)):
        train_dict = {}
        sample = train_set[i]
        sorted_translated_paths, sorted_reasoning_paths, sorted_distances = get_sorted_paths(sample)
        question = sample['question']
        train_dict[i]= {'question': question, 'translated_paths': sorted_translated_paths, 'reasoning_paths': sorted_reasoning_paths, 'path_distances': sorted_distances}
        train_dict_list.append(train_dict)
    
    
    # # val data
    val_set = RetrieverDataset(config=config, split='val')
    print ('valid_data processing')
    N = len(val_set)
    for i in tqdm(range(N)):
        val_dict = {}
        sample = val_set[i]
        sorted_translated_paths, sorted_reasoning_paths, sorted_distances = get_sorted_paths(sample)
        question = sample['question']
        val_dict[i]= {'question': question, 'translated_paths': sorted_translated_paths, 'reasoning_paths': sorted_reasoning_paths, 'path_distances': sorted_distances}
        val_dict_list.append(val_dict)
    
    
    save_path = f"data_files/{args.dataset}/processed/"
    os.makedirs(save_path, exist_ok=True)
    # save pickle to save_path
    with open(save_path + 'train_text_dict_list.pickle', 'wb') as f:
        pickle.dump(train_dict_list, f)
    with open(save_path + 'val_text_dict_list.pickle', 'wb') as f:
        pickle.dump(val_dict_list, f)
    
    
    save_path = f"data_files/{args.dataset}/processed/"

    test_set = RetrieverDataset(config=config, split='test',skip_no_path=False)
    print ('test_data processing')
    N = len(test_set)
    for i in tqdm(range(N)):
        test_dict = {}
        sample = test_set[i]
        sorted_translated_paths, sorted_reasoning_paths, sorted_distances = get_sorted_paths(sample)
        question = sample['question']
        test_dict[i]= {'question': question, 'translated_paths': sorted_translated_paths, 'reasoning_paths': sorted_reasoning_paths, 'path_distances': sorted_distances}
        test_dict_list.append(test_dict)
    with open(save_path + 'test_text_dict_list.pickle', 'wb') as f:
        pickle.dump(test_dict_list, f)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='webqsp',
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    main(args)
    
