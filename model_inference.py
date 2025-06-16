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
from src.dataset.PyG_dataset_disk import KGDataset
from src.dataset.PostProcessedDataset import ProcessedDiskDataset
from src.dataset.utils import load_pickles,decode_path,decode_topic_nodes
from termcolor import colored
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def color_print(text, color='green'):
    print(colored(text, color))


def transform_and_deduplicate_paths(decoded_results):
    """
    Transforms each path in decoded_results from a list of triplets 
    (entity, relation, entity) into a string:
        entity1 -> relation1 -> entity2 -> relation2 -> entity3 ...
    and removes duplicate paths. Also counts the total number of triplets 
    across all unique paths.

    Parameters
    ----------
    decoded_results : list
        A list where each element is of the form (path, score).
        - path is a list of triplets (entity_i, relation_i, entity_{i+1})
        - score is a float (not used in deduplication but part of the input)

    Returns
    -------
    unique_path_strings : list of str
        A list of unique arrow-formatted path strings.
        Each string looks like:
            "entity1 -> relation1 -> entity2 -> relation2 -> entity3 ..."
    total_triplets : int
        The total number of triplets in all unique paths.
    """
    seen = set()
    unique_path_strings = []
    total_triplets = 0

    for (path, score) in decoded_results:
        # Build the arrow-based chain string
        if not path:
            continue

        # Start with the first entity in the path
        chain_str = path[0][0]

        # Build "entity -> relation -> entity" chain
        for (entity1, relation, entity2) in path:
            chain_str += " -> " + relation + " -> " + entity2
        
        # Check for duplicates
        if chain_str not in seen:
            seen.add(chain_str)
            unique_path_strings.append(chain_str)
            # Add how many triplets are in this path
            total_triplets += len(path)

    return unique_path_strings, total_triplets


def write_question_paths_to_txt(question, path_strings, output_dir, i):
    """
    Writes the question and its reasoning paths to a text file in the format:
    
    Answer the question <question>, we have collected some reasoning paths to help. 
    You can use these knowledge to answer this question if you think they are helpful.

    path1
    path2
    path3

    The file is saved as: output_dir/dataset_name/sample_{i}.txt

    Parameters
    ----------
    question : str
        The natural language question being answered.
    path_strings : list of str
        A list of unique arrow-formatted path strings.
    output_dir : str
        The top-level output directory where the file will be saved.
    dataset_name : str
        The name of the dataset (used in the output file path).
    i : int
        The sample index, which is appended to the file name.
    """
    # Make sure the directory exists

    # Construct the output text
    header = (
        f"Answer the following question: \n\n {question}. \n\n we have collected some reasoning paths from a knowledge graph to help answer it. \n\n"
    )
    # Append each path on its own line
    body = "\n".join(path_strings)

    # Combine
    output_text = header + body

    # Write to file
    output_file = os.path.join(output_dir, f"sample_{i}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)


def main():
    parser = argparse.ArgumentParser(description="Train a GNN-based model with path prediction.")

    # Add arguments for Trainer configuration
    parser.add_argument("--model_type", type=str, default="GCN", help="Type of GNN model: GCN, GAT, SGConv, GIN.")
    parser.add_argument("--bidirectional", action="store_true",default=False, help="Use bidirectional GNN if set.")
    parser.add_argument("--use_stop_mlp", action="store_true",default=False, help="Use a separate MLP for stop token if set.")
    parser.add_argument("--num_mlp_layers", type=int, default=2, help="Number of MLP layers.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers.")
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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the DataLoader.")
    
    parser.add_argument("--pathTrainAfterEpoch", type=int, default=1, help="start training path loss after #epoch")
    parser.add_argument("--wandb_id", type=str, default='1')

    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    
    
    color_print(args,'cyan')
    
    # -------------------------
    # 1. load datasets
    # -------------------------    
    null_dset = KGDataset(root=f'data/{args.dataset_name}/',split=args.split)
    dset = ProcessedDiskDataset(processed_dir=f'data/post_processed_pathLabel_4o_corrected/{args.dataset_name}/',split=args.split) #! use 4o or 4o-mini as path label
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
    # 4. Infer the Model
    # -------------------------
    # **Load model states from model_dir if exists
    #! model trained using different teacher supervision
    # 2-layer
    model_dir = f"xxx.pt" # best model path
    if os.path.exists(model_dir):
        trainer.load_state_dict(torch.load(model_dir,map_location='cpu'),strict=False)
        color_print(f"Loaded model states from {model_dir}.", 'green')
    else:
        color_print(f"No model states found at {model_dir}.", 'yellow')
        sys.exit()

    trainer.to(device)
    print ('Start inference')
    retrieval_list, metadata_list = load_pickles(args['split'], null_dset)
    total_triples = 0.
    KM = [(800,1200)]
    triplet_cnt_dict = {}
    triplet_cnt_dir = f"experiments/text_from_KG/{args['wandb_id']}/{args['dataset_name']}/"
    decoding_time = []
    sample_num_triples = []
    N_dset = len(dset)
    for k,m in KM:
        save_dir = f"experiments/text_from_KG/{args['wandb_id']}/{args['dataset_name']}/K_{k}_M_{m}/GCN_bidirectional_True_numLayers_2_useStopMlp_True_numHeads_4_K_3/seed_1/"
        os.makedirs(save_dir, exist_ok=True)
        cnt = 0
        for i in tqdm(range(len(dset))):
            # if file exists continue
            if os.path.exists(os.path.join(save_dir, f"sample_{i}.txt")): 
                print (f"sample_{i}.txt exists, continue")
                continue
            question =  retrieval_list[i]['question']
            # gt_path = decode_path(dset[i], retrieval_list[i], metadata_list[i])
            # topic_nodes = decode_topic_nodes(dset[i],retrieval_list[i], metadata_list[i])
            try:
                s  = time.time()
                res = trainer.decoding(dset[i], retrieval_list[i], metadata_list[i], K=k, N=3,M=m,way='normal')
                e = time.time()
                decoding_time.append(e-s)
            except:
                continue
            # print (f"question:{question}")
            # print (f"gt paths:{gt_path}")
            res,num_triples = transform_and_deduplicate_paths(res[0])
            sample_num_triples.append(num_triples)
            
            total_triples += num_triples
            write_question_paths_to_txt(question, res, save_dir, i)
            triplet_cnt_dict[(k,m)] = total_triples
    print (f"avg decoding time:{np.mean(decoding_time)}")
    print (f"avg num triples per sample:{np.mean(sample_num_triples)}")
    # Save the triplet count dictionary to a text file
    triplet_cnt_file = os.path.join(triplet_cnt_dir, "triplet_count_dict.txt")
    with open(triplet_cnt_file, "w", encoding="utf-8") as f:
        for (k, m), count in triplet_cnt_dict.items():
            f.write(f"K={k}, M={m}: {count/N_dset} triplets\n")
    color_print(f"Saved triplet count dictionary to {triplet_cnt_file}.", 'green')


if __name__ == "__main__":
    main()
    


