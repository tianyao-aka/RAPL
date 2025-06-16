import os
import torch
from torch_geometric.data import Dataset
import pickle
from src.dataset.utils import get_topic_entity_from_path_label, extract_list_from_solution_string
from tqdm import tqdm


########################################################

class ProcessedDiskDataset(Dataset):
    def __init__(self,
                 processed_dir,
                 split,
                 raw_dataset=None,
                 labeled_topic_relation_path=None,  # Path to labeled relations from gpt4o-mini
                 retrieval_list=None,               # loaded pickle from {split}_retrieval.pkl
                 text_labels=None,                  # loaded pickle from dir of annotated labels from gpt4o
                 metadata_list=None):                # loaded pickle from {split}_metadata.pkl
        
        """
        Args:
            processed_dir (str): path where processed data is stored (and loaded).
            split (str): one of ['train','val','test'].
            raw_dataset (Dataset, optional): if first-time processing, we need the raw dataset.
            labeled_topic_relation_path (str, optional): path to a pickle for labeled relations.
            retrieval_list (List[Dict], optional): data for path matching (reasoning_paths, etc.).
            text_labels (List[List[Tuple]], optional): ground-truth path tuples for each sample.
            metadata_list (List[List[Dict]], optional): metadata_list[sample_idx] => list of dict,
                each dict contains {"h_id": ..., "relation_id": ..., "t_id": ...} for each node.
                Must align with raw_dataset and retrieval_list.
        """
        
        assert split in ['train', 'val', 'test'], "split must be 'train','val','test'."
        self.split = split
        self.split_dir = os.path.join(processed_dir, split)
        os.makedirs(self.split_dir, exist_ok=True)

        self.raw_dataset = None
        self.retrieval_list = None
        self.text_labels = None
        self.labeled_topic_relation = None
        self.metadata_list = None  # We'll store a "global" metadata_list reference
        self.min_len_indices_list =[]

        first_time = self._is_first_time()
        if first_time:
            assert raw_dataset is not None, "Raw dataset required for first-time processing."
            assert labeled_topic_relation_path is not None, (
                "Need labeled_topic_relation_path for first-time processing."
            )
            assert retrieval_list is not None, "Need retrieval_list for path matching."
            assert text_labels is not None, "Need text_labels for ground-truth path tuples."
            assert metadata_list is not None, "Need metadata_list to label line-graph nodes."

            # Store references for processing
            self.raw_dataset = raw_dataset
            self.retrieval_list = retrieval_list
            self.text_labels = text_labels
            self.metadata_list = metadata_list

            self.labeled_topic_relation = self._load_labeled_topic_relation(labeled_topic_relation_path)
            self._process_all()  # processes + saves data_.pt
        else:
            # No processing needed; we load from disk
            pass

        super().__init__(root=None)

    def _is_first_time(self):
        pt_files = [f for f in os.listdir(self.split_dir)
                    if f.startswith('data_') and f.endswith('.pt')]
        return (len(pt_files) == 0)

    def _load_labeled_topic_relation(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _process_all(self):
        """
        Iterate over raw_dataset, attach metadata_list[sample_idx],
        rebuild path_label, create topic_labels, etc., then save to disk.
        """
        for sample_idx, data_obj in enumerate(tqdm(self.raw_dataset, desc="Post Processing dataset")):
            # (A) Attach the node-level metadata for line graph nodes
            #     e.g. a list of dict, each: {"h_id":..., "relation_id":..., "t_id":...}
            #     so we can label them.

            # (B) Add new attributes (path_label, topic_labels, etc.)
            out_path = os.path.join(self.split_dir, f'data_{sample_idx}.pt')
            try:
                data_obj = self._compute_and_add_new_attr(data_obj, sample_idx)
                torch.save(data_obj, out_path)
            except Exception as e:
                print(f"Exception in sample_idx {sample_idx}: {e}")
                torch.save(data_obj, out_path)
        
        # save the min_len_indices_list and meta_list
        with open(os.path.join(self.split_dir, f'min_len_indices_list.pkl'), 'wb') as f:
            pickle.dump(self.min_len_indices_list, f,protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.split_dir, f'metadata_list.pkl'), 'wb') as f:
            pickle.dump(self.metadata_list, f,protocol=pickle.HIGHEST_PROTOCOL)


    def _compute_and_add_new_attr(self, data_obj, sample_idx):
        """
        Rebuild path_label (max_paths=10) and topic_labels for each sample.
        """
        # 1) Re-run path-labeling
        data_obj = self._rebuild_path_label(data_obj, sample_idx)

        # 2) Add or update topic_labels
        data_obj = self._update_topic_labels(data_obj, sample_idx)

        return data_obj

    def _rebuild_path_label(self, data_obj, sample_idx):
        ground_truth_tuples = self.text_labels[sample_idx]
        reasoning_paths     = self.retrieval_list[sample_idx]["reasoning_paths"]
        translated_paths    = self.retrieval_list[sample_idx]["translated_paths"]
        id2entities         = self.retrieval_list[sample_idx]["id2entities"]
        id2relations        = self.retrieval_list[sample_idx]["id2relations"]
        # max key from id2entities
        max_entity_id = max(id2entities.keys())
        max_str = id2entities[max_entity_id]
        

        # 1) Match
        matched_indices, min_len_indices = self._match_candidate_paths(
            ground_truth_tuples,
            reasoning_paths
        )
        self.min_len_indices_list.append(min_len_indices)

        # 2) Convert matched paths to triplets
        matched_paths_triplets = []
        for m_idx in matched_indices:
            if m_idx < len(translated_paths):
                path_strs = [translated_paths[m_idx]]
                triple_lists = self._paths_to_triplets(id2entities, id2relations, path_strs,maxEntStr = max_str)
                for triple_seq in triple_lists:
                    matched_paths_triplets.append(triple_seq)

        # 3) Rebuild path_label
        max_paths = 10
        num_nodes = data_obj.num_nodes
        path_label = -1 * torch.ones(num_nodes, max_paths, dtype=torch.long)

        # We stored the node-level metadata already in data_obj.metadata_list
        meta_dict = self.metadata_list[sample_idx]

        for path_idx, triple_seq in enumerate(matched_paths_triplets[:max_paths]):
            for step_i, (h, r, t) in enumerate(triple_seq):
                for node_idx in range(num_nodes):
                    if (meta_dict[node_idx]['h_id'] == h and
                        meta_dict[node_idx]['relation_id'] == r and
                        meta_dict[node_idx]['t_id'] == t):
                        path_label[node_idx, path_idx] = step_i
                        break

        data_obj.path_label = path_label
        return data_obj


    def _update_topic_labels(self, data_obj, sample_idx):
        """
        Updates the topic_labels attribute by incorporating user-labeled relations,
        path labels, and topic candidate constraints.
        """
        num_nodes = data_obj.num_nodes
        topic_labels = torch.zeros(num_nodes, dtype=torch.long)

        # (A) Extract user-labeled relations from the solution string
        response_text = self.labeled_topic_relation[sample_idx]['response_text']
        labeled_relations = extract_list_from_solution_string(response_text)

        # (B) Mark nodes as topic if their relation_text is in the user-labeled set
        meta_dict = self.metadata_list[sample_idx]
        id2rels = self.retrieval_list[sample_idx]["id2relations"]
        
        # Ensure topic_candidates is a 1D tensor of indices
        topic_candidates = data_obj.topic_candidates.squeeze()  # Shape (N,)
        candidate_indices = torch.nonzero(topic_candidates, as_tuple=True)[0]  # Indices of non-zero entries
        
        for node_idx in candidate_indices.tolist():  # Iterate only over valid candidates
            r_id = meta_dict[node_idx]['relation_id']
            rel_text = id2rels[r_id]
            if rel_text in labeled_relations:
                topic_labels[node_idx] = 1

        # (C) Additional logic: Incorporate path_label-derived topic entities
        path_label_nodes = get_topic_entity_from_path_label(data_obj.path_label, num_nodes)
        for idx in path_label_nodes:
            if idx in candidate_indices:  # Ensure it is also in topic_candidates
                topic_labels[idx] = 1

        data_obj.topic_labels = topic_labels.unsqueeze(-1)
        return data_obj


    def _match_candidate_paths(self, ground_truth_tuples, candidate_paths):
        """
        Bug-free matching of candidate paths vs. ground-truth tuples, ignoring
        extra quotes/case/spaces. Returns (matched_indices, min_length_indices).
        """
        
        def clean_str(s):
            s = s.strip()
            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                s = s[1:-1].strip()
            return s

        # Clean ground-truth
        cleaned_gt = []
        for tup in ground_truth_tuples:
            cleaned_gt.append(tuple(clean_str(x) for x in tup))

        # Clean candidate paths
        cleaned_cand = []
        for path in candidate_paths:
            cleaned_cand.append([clean_str(x) for x in path])

        if not cleaned_gt:
            return [], []

        min_length = min(len(gt) for gt in cleaned_gt)
        matched_indices = []
        min_length_indices = []

        for idx, cand_path in enumerate(cleaned_cand):
            is_matched = False
            for gt_tuple in cleaned_gt:
                if len(cand_path) == len(gt_tuple):
                    if all(
                        cand_path[i].lower().strip() == gt_tuple[i].lower().strip()
                        for i in range(len(gt_tuple))
                    ):
                        matched_indices.append(idx)
                        is_matched = True
                        break
            if not is_matched and len(cand_path) <= min_length:
                min_length_indices.append(idx)

        return matched_indices, min_length_indices

    def _paths_to_triplets(self, id2entities, id2relations, path_str_list, maxEntStr = None):
        """
        Convert a list of path strings to a list of lists of (h_id, r_id, t_id) triplets.
        """
        ent2id = {v: k for k, v in id2entities.items()}
        rel2id = {v: k for k, v in id2relations.items()}

        all_paths_triplets = []
        for p_str in path_str_list:
            segments = [seg.strip() for seg in p_str.split("->")]
            triplets_for_path = []
            for i in range(0, len(segments) - 2, 2):
                head_str = segments[i]
                rel_str  = segments[i+1]
                tail_str = segments[i+2]
                if head_str not in ent2id:
                    print(f"Entity '{head_str}' not in ent2id.")
                if rel_str not in rel2id:
                    print(f"Relation '{rel_str}' not in rel2id.")
                if tail_str not in ent2id:
                    print(f"Entity '{tail_str}' not in ent2id.")
                h_id = ent2id.get(head_str,maxEntStr)
                r_id = rel2id[rel_str]
                t_id = ent2id.get(tail_str,maxEntStr)
                triplets_for_path.append((h_id, r_id, t_id))
            all_paths_triplets.append(triplets_for_path)
        return all_paths_triplets


    def len(self):
        pt_files = [f for f in os.listdir(self.split_dir)
                    if f.startswith('data_') and f.endswith('.pt')]
        return len(pt_files)

    def get(self, idx):
        data_path = os.path.join(self.split_dir, f'data_{idx}.pt')
        data_obj = torch.load(data_path)
        return data_obj

