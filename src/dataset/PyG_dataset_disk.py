#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PygDataset(InMemoryDataset) for loading and processing custom data in PyTorch Geometric.

This dataset class expects the following directory structure under `root`:

root/
    raw/
        processed/
            train_retrieval.pkl
            val_retrieval.pkl
            test_retrieval.pkl
        gte-large-en-v1.5/
            train.pth
            val.pth
            test.pth
        annotated_paths/
            train/
                sample_0.txt
                sample_1.txt
                ...
            valid/
                sample_0.txt
                sample_1.txt
                ...
    processed/
        train_data.pt
        valid_data.pt
        test_data.pt
"""

import os
import sys
import re
import pickle
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import time
from src.dataset.utils import extract_list_from_solution_string,get_topic_entity_from_path_label


class KGDataset(Dataset):
    """
    A custom PyG dataset that DOES NOT store the entire dataset in memory.
    Instead, each sample is saved as a separate file on disk.
    """

    def __init__(self, root, split="valid", transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset is stored.
            split (str): One of ["train","valid","test"] (if you have multiple).
            transform, pre_transform (callable): Optional PyG transforms.
        """
        self.split = split
        super().__init__(root, transform, pre_transform)

        # After calling super().__init__, self.process() is invoked (if needed).
        # We'll gather the names of all sample files in self._files.

        # For example, if we stored each sample as:
        #   processed/data_0.pt, data_1.pt, data_2.pt, ...
        # we'll collect them in a list so we know how many there are.
        processed_dir = os.path.join(self.processed_dir,self.split)
        # We look for files named like "{split}_data_0.pt", etc.
        self._files = []
        # List files in the directory (train/ or valid/)
        for fname in os.listdir(processed_dir):
            file_path = os.path.join(processed_dir, fname)  # Full path to the file
            if os.path.isfile(file_path) and fname.startswith("data_") and fname.endswith(".pt"):
                self._files.append(fname)

        self._files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # sort by index
        # e.g. "valid/data_0.pt" -> index=0

    @property
    def raw_file_names(self):
        """
        Specify the raw files that must exist before processing:
          - e.g. retrieval pkl
          - embeddings pth
          - annotated path folder
        """
        return [
            os.path.join('processed',  f'{self.split}_retrieval.pkl'),
            os.path.join('gte-large-en-v1.5', f'{self.split}.pth'),
            os.path.join("annotated_paths_GPT4o", self.split)
        ]

    @property
    def processed_file_names(self):
        """
        Instead of one big file, we have many: e.g. "train_data_0.pt" ... "train_data_N.pt"
        But PyG needs *some* reference. We could just return an empty list or
        a single placeholder. We'll return an empty list to let PyG know that
        we handle enumerating processed files ourselves.
        """
        
        # return ['train/']
        return ['train/','val/','test/']

    def download(self):
        # No downloading; we assume raw data is provided locally
        pass

    def process(self):
        """
        Process the raw data for this split and save each sample as a separate .pt file.
        For instance:
          processed/
            valid_data_0.pt
            valid_data_1.pt
            ...
        Also store metadata and min_len_indices separately if you like.
        """
        
        # 1) Load your raw data, embeddings, etc.
        print ('self.raw_dir:',self.raw_dir)
        retrieval_path = os.path.join(self.raw_dir, "processed", f"{self.split}_retrieval.pkl")
        emb_path       = os.path.join(self.raw_dir, "gte-large-en-v1.5", f"{self.split}.pth")
        label_path     = os.path.join(self.raw_dir, "annotated_paths_GPT4o", self.split)

        if not os.path.exists(retrieval_path) or not os.path.exists(emb_path):
            print (retrieval_path)
            print (emb_path)
            print(f"[Warning] Missing raw data for {self.split}, skipping process.")
            sys.exit(1)
            return
        print ('loading pickle files')
        s = time.time()
        with open(retrieval_path, "rb") as f:
            retrieval_data = pickle.load(f)  # e.g. List[Dict]
        print ('time cost:',time.time()-s)
        print ('loading tensor to cpu')
        emb_data = torch.load(emb_path,map_location='cpu') 
        print ('collecting text labels')
        text_labels = None
        if os.path.isdir(label_path):
            text_labels = self._collect_annotated_paths(label_path)


        # 2) Build the processed Data objects sample by sample
        #    We do NOT store them in a single list (to avoid OOM).
        #    Instead, process each item -> save to disk immediately.
        self.min_len_indices = []
        metadata_list = []

        out_dir = os.path.join(self.processed_dir, self.split)
        os.makedirs(out_dir, exist_ok=True)

        for idx, sample in enumerate(tqdm(retrieval_data)):
            # For demonstration, we can limit if data is huge:
            # if idx > 10: break  # optional

            data_obj, meta_data, min_len_indices_sample = self._process_single_sample(
                sample, emb_data, text_labels, idx
            )
            # Keep track for saving later if desired
            self.min_len_indices.append(min_len_indices_sample)
            metadata_list.append(meta_data)

            # Save the Data object to a file, e.g.: "valid_data_0.pt"
            out_path = os.path.join(out_dir, f"data_{idx}.pt")
            torch.save(data_obj, out_path)

        # 3) Save min_len_indices and metadata if desired
        with open(os.path.join(out_dir, f"{self.split}_min_len_indices.pkl"), "wb") as f:
            pickle.dump(self.min_len_indices, f)

        with open(os.path.join(out_dir, f"metadata_{self.split}.pkl"), "wb") as f:
            pickle.dump(metadata_list, f)

    def _process_single_sample(self, sample, emb_data, text_labels, idx):
        """
        Convert ONE sample from raw data into a PyG `Data` object, plus metadata and min_len_indices.

        Args:
            sample (Dict): A dictionary describing one sample, typically from retrieval_data[idx].
            emb_data (Dict[int, Dict[str, Tensor]]): Mapped from sample['id'] -> embedding dicts.
            text_labels (List[List[tuple]] or None): If not None, text_labels[idx] gives
                ground-truth path tuples for this sample.
            idx (int): The index of this sample in retrieval_data, used to look up text_labels, etc.

        Returns:
            data_obj (Data): A single PyG Data object representing this sample's line graph.
            meta_data (List[Dict]): A list of metadata for each line-graph node,
                                    e.g. {'relation_id', 'h_id', 't_id'}.
            min_len_indices_sample (List[int]): The "min_length_indices" from path matching.
        """

        # ----------------------------
        # 1) Extract sample fields
        # ----------------------------
        s_id   = sample['id']
        nx_g   = sample['nx_graph']          # The original knowledge graph in NetworkX
        h_list = sample['h_id_list']
        t_list = sample['t_id_list']

        q_entity_id_list = sample['q_entity_id_list']
        reasoning_paths  = sample['reasoning_paths']
        translated_paths = sample['translated_paths']

        # If the raw data has ID->entities/relations, we might invert them for path parsing:
        id2entities  = sample['id2entities']
        id2relations = sample['id2relations']
        

        # ----------------------------
        # 2) Retrieve embeddings
        # ----------------------------
        # Each sample's embedding info is stored under emb_data[s_id]
        if s_id not in emb_data:
            raise KeyError(f"Sample id={s_id} not found in emb_data.")

        emb_dict = emb_data[s_id]
        q_emb         = emb_dict['q_emb']
        entity_embs   = emb_dict['entity_embs']   # shape [num_entities, entity_dim]
        relation_embs = emb_dict['relation_embs'] # shape [num_relations, relation_dim]

        # ----------------------------
        # 3) Match ground truth paths (if text_labels is not None)
        # ----------------------------
        min_len_indices_sample = []
        matched_paths_triplets = []

        if text_labels is not None:
            # text_labels[idx] is a list of ground-truth path tuples, e.g. [('a.b.c','d.e.f'), ...]
            ground_truth_tuples = text_labels[idx]  # => the ground-truth for this sample

            # 3a) Attempt to match them with the candidate reasoning paths
            matched_indices, min_length_indices = self._match_candidate_paths(
                ground_truth_tuples, reasoning_paths
            )
            min_len_indices_sample = min_length_indices  # store for later

            # 3b) For each matched index, parse the translated_paths[m_idx] to triplets
            for m_idx in matched_indices:
                if m_idx < len(translated_paths):
                    path_str = translated_paths[m_idx]
                    path_list = [path_str]  # wrap single string
                    triplets_per_path = self._paths_to_triplets(id2entities, id2relations, path_list)
                    # _paths_to_triplets returns List[List[ (h_id, r_id, t_id) ]]
                    for triple_seq in triplets_per_path:
                        matched_paths_triplets.append(triple_seq)
        else:
            # If text_labels is None, we skip path matching entirely
            pass

        # ----------------------------
        # 4) Build the line graph
        # ----------------------------
        # This is the single-sample version of your `_build_line_graph()`.
        data_obj, meta_data = self._build_line_graph(
            nx_graph           = nx_g,
            relation_embs      = relation_embs,
            h_id_list          = h_list,
            t_id_list          = t_list,
            entity_embs        = entity_embs,
            text_entity_list   = sample.get('text_entity_list', []),
            use_entity_feature = True,
            all_matched_triplets = matched_paths_triplets
        )

        # ----------------------------
        # 5) Annotate data_obj
        # ----------------------------
        # e.g. store question embedding
        data_obj.q_emb = q_emb

        # Mark nodes whose head-entity is in q_entity_id_list
        question_ent_id_set = set(q_entity_id_list)
        num_nodes = data_obj.num_nodes
        topic_candidates = torch.zeros(num_nodes, dtype=torch.long)
        gpt_labeled_topic_candidates = torch.zeros(num_nodes, dtype=torch.long)
        for node_idx in range(num_nodes):
            h_ent = meta_data[node_idx]['h_id']
            if h_ent in question_ent_id_set:
                topic_candidates[node_idx] = 1
        topic_indices = get_topic_entity_from_path_label(data_obj.path_label,data_obj.x.shape[0])
        for t in topic_indices:
            gpt_labeled_topic_candidates[t] = 1
        data_obj.topic_candidates = topic_candidates.unsqueeze(-1)  # shape [num_nodes, 1]

        # data_obj is now ready for saving/returning
        return data_obj, meta_data, min_len_indices_sample


    # ------------------------------------------------------------------
    # Example line-graph building method
    # ------------------------------------------------------------------
    def _build_line_graph(self,
                          nx_graph,
                          relation_embs,
                          h_id_list,
                          t_id_list,
                          entity_embs=None,
                          text_entity_list=None,
                          use_entity_feature=True,
                          all_matched_triplets=None):

        """
        Construct a line graph from a directed NetworkX graph (nx_graph).
        Each edge of nx_graph becomes a node in the line graph, and two line-graph nodes
        are connected if their original edges in nx_graph share a common vertex.

        Additionally, we now take 'all_matched_triplets' (a list of matched paths,
        each path is a list of (h, r, t) triplets) and use it to label line-graph nodes.

        Returns:
            (Data, dict): A tuple containing:
                - Data: A PyTorch Geometric Data object with:
                    - x: node feature matrix for the line graph
                    - edge_index: connectivity of the line graph
                    - metadata: dictionary attached as an attribute
                    - adj_1hop: 1-hop adjacency (sparse PyTorch)
                    - adj_2hop: 2-hop adjacency (sparse PyTorch)
                    - path_label: (num_nodes, 6) LongTensor with path-step labels,
                      or -1 if node not in that step.
                - metadata: dict that maps node_index -> {relation_id, h_id, t_id}
        """

        line_g = nx.line_graph(nx_graph)
        line_node_map = {}
        line_node_features = []
        metadata_list = []

        entity_emb_dim = entity_embs.shape[1] if entity_embs is not None else 0
        relation_emb_dim = relation_embs.shape[1] if relation_embs is not None else 0

        idx_counter = 0
        for (src_edge, dst_edge) in line_g.nodes():
            edge_data = nx_graph[src_edge][dst_edge]
            triple_id = edge_data['triple_id']
            relation_id = edge_data['relation_id']

            rel_emb = relation_embs[relation_id]
            h_idx   = h_id_list[triple_id]
            t_idx   = t_id_list[triple_id]

            # metadata
            metadata_list.append({
                'relation_id': relation_id,
                'h_id': h_idx,
                't_id': t_idx
            })

            # optional: use entity features
            if use_entity_feature and (entity_embs is not None):
                h_emb = entity_embs[h_idx] if 0 <= h_idx < entity_embs.shape[0] else torch.zeros(entity_emb_dim)
                t_emb = entity_embs[t_idx] if 0 <= t_idx < entity_embs.shape[0] else torch.zeros(entity_emb_dim)
                node_feature = torch.cat([h_emb, rel_emb, t_emb], dim=0)
            else:
                node_feature = rel_emb

            line_node_map[(src_edge, dst_edge)] = idx_counter
            line_node_features.append(node_feature)
            idx_counter += 1

        num_nodes = len(line_node_features)
        edge_index_list = []
        for (n1, n2) in line_g.edges():
            src_idx = line_node_map[n1]
            dst_idx = line_node_map[n2]
            edge_index_list.append([src_idx, dst_idx])

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if num_nodes > 0:
            x = torch.stack(line_node_features, dim=0)
        else:
            feat_dim = (2 * entity_emb_dim + relation_emb_dim) if use_entity_feature else relation_emb_dim
            x = torch.empty((0, feat_dim), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        one_hop_neighbors = self._get_1hop_neighbors(edge_index, num_nodes)
        data.one_hop_neighbors = one_hop_neighbors  # list of lists

        # label matched triplets
        max_paths = 6
        path_label = -1 * torch.ones(num_nodes, max_paths, dtype=torch.long)
        if all_matched_triplets and num_nodes > 0:
            for path_idx, path_triplets in enumerate(all_matched_triplets[:max_paths]):
                for step_i, (h, r, t) in enumerate(path_triplets):
                    for node_idx in range(num_nodes):
                        meta = metadata_list[node_idx]
                        if (meta['h_id'] == h and 
                            meta['relation_id'] == r and 
                            meta['t_id'] == t):
                            path_label[node_idx, path_idx] = step_i
                            break
        data.path_label = path_label

        return data, metadata_list

    # ------------------------------------------------------------------
    # The following methods are the same as in your code:
    # _collect_annotated_paths, _parse_rational_paths, etc.
    # ------------------------------------------------------------------

    def _collect_annotated_paths(self, folder_path):
        if not os.path.isdir(folder_path):
            return None

        txt_files = [f for f in os.listdir(folder_path)
                     if f.startswith("sample_") and f.endswith(".txt")]
        def extract_index(fname):
            base = fname.replace(".txt", "")
            parts = base.split("_")
            return int(parts[-1])
        txt_files_sorted = sorted(txt_files, key=extract_index)

        parsed_paths_list = []
        for txt_file in txt_files_sorted:
            parsed = self._parse_rational_paths(folder_path, txt_file)
            parsed_paths_list.append(parsed)
        return parsed_paths_list

    def _parse_rational_paths(self, file_dir, file_name):
        
        """
        Given a directory (file_dir) and a file name (file_name) that points to a text file
        (e.g. file_dir='path/to' and file_name='sample_1.txt'), this function reads and parses
        all lines after the phrase "The rational paths are:". Each path line in the file is
        expected to follow this format:

            1. [location.location.partially_contains, geography.river.basin_countries]
            2. [location.location.time_zones]
            3. [other.path.example]

        The function extracts each bracketed sequence, splits by commas, and creates a tuple
        of strings. It then collects these tuples into a list and returns it.
        """
        
        full_path = os.path.join(file_dir, file_name)
        lines = []
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        parsed_paths = []
        in_section = False
        pattern = re.compile(r'^\s*\d+\.\s*\[(.*?)\]\s*$')
        for line in lines:
            if "The rational paths are:" in line:
                in_section = True
                continue
            if in_section:
                match = pattern.match(line.strip())
                if match:
                    bracket_content = match.group(1)
                    parts = [p.strip() for p in bracket_content.split(',')]
                    parsed_paths.append(tuple(parts))
        return parsed_paths


    def _match_candidate_paths(self,ground_truth_tuples, candidate_paths):
        """
        Given:
            ground_truth_tuples : list of tuples, e.g. [('a.b.c','d.e.f'), ('g.h','i.j')]
            candidate_paths     : list of lists, each a list of strings

        We say a candidate path matches a ground-truth tuple if:
            - They have the same length
            - Each component matches ignoring case/spaces

        Returns:
            (matched_indices, min_length_indices)

            matched_indices:
                Indices in candidate_paths that match at least one ground_truth_tuple
            min_length_indices:
                Indices in candidate_paths whose length <= minimum ground_truth_tuple length
                but are NOT in matched_indices.
        """

        def clean_str(s):
            """
            Remove surrounding single/double quotes if present, and strip whitespace.
            """
            s = s.strip()
            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                s = s[1:-1].strip()
            return s

        # 1) Normalize ground_truth_tuples: each entry is a tuple of strings
        fixed_ground_truth = []
        for tup in ground_truth_tuples:
            cleaned_tuple = tuple(clean_str(x) for x in tup)
            fixed_ground_truth.append(cleaned_tuple)

        # 2) Normalize candidate_paths: each entry is a list of strings
        fixed_candidate_paths = []
        for path in candidate_paths:
            cleaned_path = [clean_str(x) for x in path]
            fixed_candidate_paths.append(cleaned_path)

        # Update references to the cleaned versions
        ground_truth_tuples = fixed_ground_truth
        candidate_paths = fixed_candidate_paths

        # If no ground-truth tuples, return empty
        if not ground_truth_tuples:
            return [], []

        # Find the shortest ground-truth tuple length
        min_length = min(len(gt) for gt in ground_truth_tuples)

        matched_indices = []
        min_length_indices = []

        # Check each candidate path
        for idx, cand_path in enumerate(candidate_paths):
            is_matched = False

            # Compare against each ground-truth tuple
            for gt_tuple in ground_truth_tuples:
                if len(cand_path) == len(gt_tuple):
                    # Compare element by element, ignoring case/spaces
                    if all(
                        cand_path[i].strip().lower() == gt_tuple[i].strip().lower()
                        for i in range(len(gt_tuple))
                    ):
                        matched_indices.append(idx)
                        is_matched = True
                        break

            # If not matched, but length <= min_length, mark it
            if not is_matched and len(cand_path) <= min_length:
                min_length_indices.append(idx)

        return matched_indices, min_length_indices



    def _paths_to_triplets(self, id2entities, id2relations, paths):
        """
        Given a list of path strings, each describing a path in the form:
          "Entity -> Relation -> Entity -> Relation -> Entity -> ..."

        Returns a list of lists of (head_id, relation_id, tail_id) triplets.
        If an entity or relation string is missing, raises KeyError.
        """
        
        ent2id = {v:k for k,v in id2entities.items()}
        rel2id = {v:k for k,v in id2relations.items()}

        all_paths_triplets = []
        for path_str in paths:
            segments = [seg.strip() for seg in path_str.split("->")]
            triplets_for_this_path = []
            for i in range(0, len(segments)-2, 2):
                head_str = segments[i]
                rel_str  = segments[i+1]
                tail_str = segments[i+2]
                if head_str not in ent2id:
                    raise KeyError(f"Entity '{head_str}' not found in ent2id.")
                if rel_str not in rel2id:
                    raise KeyError(f"Relation '{rel_str}' not found in rel2id.")
                if tail_str not in ent2id:
                    raise KeyError(f"Entity '{tail_str}' not found in ent2id.")
                h_id = ent2id[head_str]
                r_id = rel2id[rel_str]
                t_id = ent2id[tail_str]
                triplets_for_this_path.append((h_id, r_id, t_id))
            all_paths_triplets.append(triplets_for_this_path)
        return all_paths_triplets


    def _get_1hop_neighbors(self, edge_idx, num_nodes):
        """
        Given edge_idx of shape [2, E], build a list of neighbors for each node.
        """
        neighbors_dict = {i: set() for i in range(num_nodes)}
        for src, dst in edge_idx.t().tolist():
            neighbors_dict[src].add(dst)
            neighbors_dict[dst].add(src)
        return [list(neighbors_dict[i]) for i in range(num_nodes)]

    def len(self):
        """
        Total number of samples = number of .pt files we created in 'process()'.
        """
        return len(self._files)

    def get(self, idx):
        """
        Load the idx-th sample from disk and return it as a PyG `Data` object.
        """
        fname = self._files[idx]
        fpath = os.path.join(self.processed_dir,self.split, fname)
        data_obj = torch.load(fpath)
        return data_obj
    