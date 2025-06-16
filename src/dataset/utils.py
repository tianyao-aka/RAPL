import os
import pickle
import torch
from torch_geometric.data import Data

import ast
import re
from termcolor import colored



def get_topic_entity_from_path_label(path_label,num_nodes):
    N_path = path_label.size(1)
    # Extract topic-entity ground-truths (valid paths)
    valid_columns = []
    step_to_node = {}
    step_to_node[0] = []
    for c in range(N_path):  # iterate over 6 path columns
        column_vals = path_label[:, c]  # shape [num_nodes_in_graph]
        if torch.any(column_vals == 0):  # if any node in column has step==0
            valid_columns.append(c)

    if len(valid_columns) == 0:
        return []
    
    for c in valid_columns:
        column_vals = path_label[:, c]  # shape [num_nodes_in_graph]

        for i_node in range(num_nodes):
            step = column_vals[i_node].item()
            if step == 0:
                step_to_node[step].append(i_node)  # Align to batch index
    return step_to_node[0]


def extract_list_from_solution_string(solution_str: str) -> list:
    """
    Given a string in the format:
    '<Solution:> [law.inventor.inventions, base.argumentmaps.innovator.original_ideas]'
    
    This function extracts the list of dot-separated items inside the brackets and returns it as a Python list.
    
    Example input:
    '<Solution:> [law.inventor.inventions, base.argumentmaps.innovator.original_ideas]'
    
    Example output:
    ['law.inventor.inventions', 'base.argumentmaps.innovator.original_ideas']
    
    Parameters:
    solution_str (str): The input string to parse.
    
    Returns:
    list: A list of strings extracted from within the brackets.
    """
    # Use regular expression to extract the part inside square brackets
    match = re.search(r'\[(.*?)\]', solution_str)
    if match:
        # Split by comma and strip any leading/trailing spaces
        elements = [item.strip() for item in match.group(1).split(',')]
        return elements
    else:
        # If no valid format found, return an empty list
        return []


def load_pickles(split, dataset):
    """
    Load the necessary pickle files for a given split. This function checks if a processed retrieval file exists.
    If the file does not exist, it falls back to loading a raw retrieval file, processes it into the required format,
    and saves the processed retrieval file.

    Parameters
    ----------
    split : str
        The dataset split to load (e.g., 'train', 'dev', 'test').

    dataset : Any
        An object that has 'raw_dir' and 'processed_dir' attributes,
        representing directories where data is stored.

    Returns
    -------
    retrieval_list : list of dict
        A list of dictionaries loaded from {raw_dir}/processed/{split}_retrieval.pkl
        (or generated from {raw_dir}/processed/{split}_id_entity_mapping.pkl if necessary),
        typically containing 'question', 'id2entities', and 'id2relations'.

    metadata_list : list of dict
        A list of dictionaries loaded from {processed_dir}/{split}/metadata_{split}.pkl,
        where each element has keys: 'relation_id', 'h_id', and 't_id'.
    """
    # File paths
    raw_retrieval_file = os.path.join(dataset.raw_dir, 'processed', f'{split}_retrieval.pkl')
    retrieval_file = os.path.join(dataset.raw_dir, 'processed', f'{split}_id_entity_mapping.pkl')
    metadata_file = os.path.join(dataset.processed_dir, split, f'metadata_{split}.pkl')

    # Step 1: Load retrieval data (process from raw file if necessary)
    if os.path.exists(retrieval_file):
        print ("load from id_entity mapping file")
        # Load directly if processed file already exists
        with open(retrieval_file, 'rb') as f:
            retrieval_list = pickle.load(f)
    else:
        # Load from raw file and process into required format
        with open(raw_retrieval_file, 'rb') as f:
            raw_retrieval_list = pickle.load(f)  # list of dicts

        # Process each item into the desired format
        retrieval_list = []
        for entry in raw_retrieval_list:
            processed_entry = {
                "question": entry.get("question"),
                "id2entities": entry.get("id2entities"),
                "id2relations": entry.get("id2relations")
            }
            retrieval_list.append(processed_entry)

        # Save processed retrieval list to retrieval_file using highest protocol
        print ('No id_entity mapping file found, pickle dump')
        with open(retrieval_file, 'wb') as f:
            pickle.dump(retrieval_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Step 2: Load metadata
    with open(metadata_file, 'rb') as f:
        metadata_list = pickle.load(f)  # list of dicts

    return retrieval_list, metadata_list



def decode_topic_nodes(data,retrieval_dict, metadata_list):
    nodes = []
    non_zero_indices = (data.topic_labels != 0).nonzero(as_tuple=True)[0]
    non_zero_indices = non_zero_indices.tolist()
    id2entities = retrieval_dict['id2entities']
    id2relations = retrieval_dict['id2relations']
    for n in non_zero_indices:
        triple_info = metadata_list[n]  # {'relation_id', 'h_id', 't_id'}

        r_id = triple_info['relation_id']
        h_id = triple_info['h_id']
        t_id = triple_info['t_id']
        h_text = id2entities[h_id]
        r_text = id2relations[r_id]
        t_text = id2entities[t_id]
        nodes.append((h_text, r_text, t_text))
    return nodes


def decode_path(data, retrieval_dict, metadata_list):
    """
    Decode the path_label in the Data object to yield paths as textual triples.

    Parameters
    ----------
    data : torch_geometric.data.Data
        A PyG Data object that must contain:
          - data.path_label of shape (N_nodes, N_paths).
            For each column (i.e., path), row r can be -1 (meaning the node at row r
            is not in this path) or a non-negative integer representing the step
            index of that node in the path.

    retrieval_list : list of dict
        The list of dictionaries loaded from {split}_retrieval.pkl.
        Each index corresponds to the same node index used in `metadata_list`.
        Each element is expected to contain 'id2entities' and 'id2relations' 
        for textual mapping.

    metadata_list : list of dict
        The metadata loaded from metadata_{split}.pkl, where each index 
        corresponds to a node in the graph. An example entry is:
            {
                'relation_id': r_id,
                'h_id': h_idx,
                't_id': t_idx
            }

    Returns
    -------
    list_of_paths : list
        A list of decoded paths. Each element of this list corresponds to 
        one valid path (one column in path_label). Each path is a list of 
        triples in text form: (head_text, relation_text, tail_text), in the 
        correct step order: step 0, then step 1, step 2, etc.
    """
    path_label = data.path_label  # shape: (N_nodes, N_paths)
    num_nodes, num_cols = path_label.shape

    
    list_of_paths = []

    for col in range(num_cols):
        col_data = path_label[:, col]

        # Gather (row_idx, step_val) for rows where step_val != -1
        valid_steps = []
        for row_idx in range(num_nodes):
            step_val = col_data[row_idx].item()
            if step_val != -1:
                valid_steps.append((row_idx, step_val))

        # If no valid steps, skip
        if not valid_steps:
            continue

        # Sort by step_val in ascending order, so step 0 -> step 1 -> step 2 -> ...
        valid_steps.sort(key=lambda x: x[1])

        # Decode each node in the sorted path
        path_triples = []
        for node_idx, step_val in valid_steps:
            triple_info = metadata_list[node_idx]  # {'relation_id', 'h_id', 't_id'}

            r_id = triple_info['relation_id']
            h_id = triple_info['h_id']
            t_id = triple_info['t_id']

            id2entities = retrieval_dict['id2entities']
            id2relations = retrieval_dict['id2relations']

            h_text = id2entities[h_id]
            r_text = id2relations[r_id]
            t_text = id2entities[t_id]

            path_triples.append((h_text, r_text, t_text))

        list_of_paths.append(path_triples)

    return list_of_paths

def decode_path_from_indices(node_indices, retrieval_dict, metadata_list):
    """
    Decode the given list of node indices to yield paths as textual triples.

    Parameters
    ----------
    node_indices : list of list of int
        A list where each element is a list of node indices representing a path.

    retrieval_dict : dict
        A dictionary containing 'id2entities' and 'id2relations' for textual mapping.

    metadata_list : list of dict
        The metadata loaded from metadata_{split}.pkl, where each index 
        corresponds to a node in the graph. An example entry is:
            {
                'relation_id': r_id,
                'h_id': h_idx,
                't_id': t_idx
            }

    Returns
    -------
    list_of_paths : list
        A list of decoded paths. Each element of this list corresponds to 
        one valid path. Each path is a list of triples in text form: 
        (head_text, relation_text, tail_text).
    """
    list_of_paths = []

    for path in node_indices:
        path_triples = []
        for node_idx in path:
            triple_info = metadata_list[node_idx]  # {'relation_id', 'h_id', 't_id'}

            r_id = triple_info['relation_id']
            h_id = triple_info['h_id']
            t_id = triple_info['t_id']

            id2entities = retrieval_dict['id2entities']
            id2relations = retrieval_dict['id2relations']

            h_text = id2entities[h_id]
            r_text = id2relations[r_id]
            t_text = id2entities[t_id]

            path_triples.append((h_text, r_text, t_text))

        list_of_paths.append(path_triples)

    return list_of_paths
