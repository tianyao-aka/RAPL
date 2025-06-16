import networkx as nx
import numpy as np
import os
import sys
import pickle
import torch
import torch.nn.functional as F

from tqdm import tqdm

class RetrieverDataset:
    def __init__(
        self,
        config,
        split,
        skip_no_path=True
    ):
        # init nx graph list
        self.nx_graphs = []
        
        # Load pre-processed data.
        dataset_name = config['dataset']['name']
        
        RetrieverDatasetPath = f'data_files/{dataset_name}/processed/{split}_retrieval.pkl'
        if os.path.exists(RetrieverDatasetPath):
            with open(RetrieverDatasetPath, 'rb') as f:
                self.processed_dict_list = pickle.load(f)
            # self.emb_dict = self._load_emb(
            #                 dataset_name, config['dataset']['text_encoder_name'], split)
            print ('Loaded processed_dict_list from pkl')
        else:
            processed_dict_list = self._load_processed(dataset_name, split)
            # Extract directed shortest paths from topic entities to answer
            # entities or vice versa as weak supervision signals for triple scoring.
            triple_score_dict = self._get_triple_scores(
                dataset_name, split, processed_dict_list)
            # -------------------------------------------------------------
            # NEW: extract and translate paths/relations
            # -------------------------------------------------------------
            path_relation_dict = self._get_translated_paths_and_relations(
                dataset_name, split, processed_dict_list
            )
            # -------------------------------------------------------------

            # Load pre-computed embeddings.
            # emb_dict = self._load_emb(
            #     dataset_name, config['dataset']['text_encoder_name'], split)
            # self.emb_dict = emb_dict
            emb_dict=None
            self.emb_dict = emb_dict
            # Put everything together.
            self._assembly(
                processed_dict_list,
                triple_score_dict,
                emb_dict,
                skip_no_path,
                path_relation_dict  # NEW input
            )
            # save the processed_dict_list to pkl
            try:
                os.makedirs(os.path.dirname(RetrieverDatasetPath), exist_ok=True)
                print ('pickle with highest protocol')
                with open(RetrieverDatasetPath, 'wb') as f:
                    pickle.dump(self.processed_dict_list, f,protocol=pickle.HIGHEST_PROTOCOL)
                print (f"Saved processed_dict_list to {RetrieverDatasetPath}")
            except Exception as e:
                print (e)

    def _load_processed(
        self,
        dataset_name,
        split
    ):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)
        
    def _load_processed_retrieval(
        self,
        dataset_name,
        split
    ):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}_retrieval.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list
    ):
        save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            print (f"Loading triple_score_dict from {save_file}")
            return torch.load(save_file)

        triple_score_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i = self._extract_paths_and_score(
                sample_i
            )
            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }
        torch.save(triple_score_dict, save_file)
        
        return triple_score_dict

    def _extract_paths_and_score(
        self,
        sample
    ):
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs.
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # Convert raw entity-based path to triple-based path.
        path_list = []
        for path in path_list_:
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)
            triples_path = []
            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                triple_id_i_list = [nx_g[h_id_i][t_id_i]['triple_id']]
                triples_path.append(triple_id_i_list)
            path_list.append(triples_path)

        num_triples = len(sample['h_id_list'])
        triple_scores = self._score_triples(path_list, num_triples)
        
        return triple_scores, max_path_length

    def _get_nx_g(
        self,
        h_id_list,
        r_id_list,
        t_id_list
    ):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)
        self.nx_graphs.append(nx_g)
        return nx_g

    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        """
        Returns all shortest paths (in edges) between q_entity_id and a_entity_id
        in both forward and backward directions.
        """
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        
        # Only keep minimal ones if both directions exist
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        return refined_paths

    def _score_triples(
        self,
        path_list,
        num_triples
    ):
        triple_scores = torch.zeros(num_triples)
        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.0
        return triple_scores

    def _load_emb(
        self,
        dataset_name,
        text_encoder_name,
        split
    ):
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        dict_file = torch.load(file_path)
        return dict_file

    # -------------------------------------------------------------------------
    # NEW CODE STARTS HERE
    # -------------------------------------------------------------------------

    def cut_off_paths(self, nx_g, q_entity_id, a_entity_id, tolerance=2):
        """
        This function computes the shortest path length d from q_entity_id
        to a_entity_id (one direction only, no reverse). Then uses
        nx.all_simple_paths to collect all simple paths whose length in edges
        is between d and d + tolerance.

        Parameters
        ----------
        nx_g : nx.DiGraph
            The directed graph.
        q_entity_id : int
            Source entity ID.
        a_entity_id : int
            Target entity ID.
        tolerance : int
            Allowed extra path length beyond the shortest path.

        Returns
        -------
        list_of_paths : list
            A list of paths (each path is a list of node-IDs) whose edge-length
            is in [d, d + tolerance].
        """
        try:
            # Get a single shortest path (one-direction only, no backward).
            shortest_paths = nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id)
            shortest_paths = list(shortest_paths)
            if len(shortest_paths)>50 or len(shortest_paths[0])>5:
                return shortest_paths
        except:
            return []  # no path from q_entity_id to a_entity_id

        # Among all shortest paths from q_entity_id -> a_entity_id, pick the length
        # of the first one (any would do, they are all minimal).
        if len(shortest_paths) == 0:
            return []
        min_len = len(shortest_paths[0]) - 1  # number of edges

        # Now gather all simple paths with edge-length up to min_len + tolerance
        cutoff = min_len + tolerance
        all_paths_within_cutoff = nx.all_simple_paths(
            nx_g,
            source=q_entity_id,
            target=a_entity_id,
            cutoff=cutoff
        )
        # Filter out those that have edges < min_len or edges > min_len + tolerance
        list_of_paths = []
        for path in all_paths_within_cutoff:
            length_in_edges = len(path) - 1
            if min_len <= length_in_edges <= (min_len + tolerance):
                list_of_paths.append(path)

        return list_of_paths

    def tranlate_paths(self, nx_g, paths, sample):
        """
        This function translates a list of paths (each path is a list of node-IDs)
        into textual format and collects related relation-IDs.

        Parameters
        ----------
        nx_g : nx.DiGraph
            The directed graph.
        paths : list of lists
            A list of paths, each path is a list of entity-IDs (e.g. [1, 5, 2]).
        sample : dict
            A processed sample containing 'id2entities' and 'id2relations',
            among other fields.

        Returns
        -------
        translated_paths : list of str
            A list of paths in textual form, e.g. for path [1,5,2],
            "entity1 -> relationX -> entity5 -> relationY -> entity2"
        reasoning_paths : list of list
            A list of only the relations in textual form per path, e.g.
            for [1, 5, 2] => ["relationX", "relationY"].
        distances : list of int
            A list of path lengths in edges. E.g. if paths=[[1,5,2],[1,7,4,9]],
            then distances=[2,3].
        relation_ids_tensor : list of torch.Tensor
            A list of variable-length Tensors containing relation IDs
            for each path. You could pad them if you need a single 2D tensor.
        """
        
        id2entity = sample['id2entities']
        id2relation = sample['id2relations']

        translated_paths = []
        reasoning_paths = []
        distances = []
        relation_ids = []  # will be a list of lists

        for path in paths:
            # Number of edges = number of "h->t" transitions
            path_len = len(path) - 1
            distances.append(path_len)

            # Collect textual expansions and relation IDs
            path_text = []
            rels_text = []
            current_rel_ids = []

            # Initialize the textual path with the first entity
            if path:  # ensure path is not empty
                path_text.append(id2entity[path[0]])

            for i in range(path_len):
                h_id = path[i]
                t_id = path[i+1]
                rel_id = nx_g[h_id][t_id]['relation_id']

                # Append the relation text
                relation_text = id2relation[rel_id]
                path_text.append(relation_text)  # "-> relation ->"
                rels_text.append(relation_text)

                current_rel_ids.append(rel_id)

                # Append the tail entity text so we see each intermediate node
                path_text.append(id2entity[t_id])  # "-> entity_t ->"

            # Convert path_text into a single string
            # e.g. "Justin Bieber -> film.producer.film -> someEntity -> film.actor.film -> ..."
            translated_paths.append(" -> ".join(path_text))

            # For reasoning_paths, we only keep the relations
            # If you want more detail (including entities), you can store them here instead.
            reasoning_paths.append(rels_text)

            # Append the current path's relation IDs to the big list
            relation_ids.append(current_rel_ids)

        # Convert each list of relation IDs to a Tensor.
        # If you need a single 2D tensor, you can pad them first.
        relation_ids_tensor = [torch.tensor(rids) for rids in relation_ids]
        # print (translated_paths)
        # print (reasoning_paths)
        # print (distances)
        return translated_paths, reasoning_paths, distances, relation_ids_tensor
    
    
    def _extract_translated_path_and_relations(self, sample, tolerance=2):
        """
        Similar to _extract_paths_and_score, but uses the newly defined
        cut_off_paths and tranlate_paths to gather translated results.
        """
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # We'll collect all paths from q_entity -> a_entity with tolerance cutoff
        all_paths = []
        for q_id in sample['q_entity_id_list']:
            for a_id in sample['a_entity_id_list']:
                # Only forward direction, as per your requirement.
                # This yields paths in [d, d + tolerance].
                forward_paths = self.cut_off_paths(nx_g, q_id, a_id, tolerance)
                if len(forward_paths) > 0:
                    all_paths.extend(forward_paths)
        if len(all_paths) == 0:
            # No valid paths found, return default empty.
            return [], [], [], []

        # Translate them
        translated_paths, reasoning_paths, distances, relation_ids_tensor = \
            self.tranlate_paths(nx_g, all_paths, sample)

        return translated_paths, reasoning_paths, distances, relation_ids_tensor

    def _get_translated_paths_and_relations(
        self,
        dataset_name,
        split,
        processed_dict_list
    ):
        """
        Similar to _get_triple_scores, but calls _extract_translated_path_and_relations
        for each sample and stores the resulting 4 returns in path_relation_dict.
        """
        # You can optionally save to disk if you want, here we simply compute them.
        path_relation_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            tpaths, rpaths, distances, rel_ids = self._extract_translated_path_and_relations(
                sample_i, tolerance=2
            )
            path_relation_dict[sample_i_id] = {
                'translated_paths': tpaths,
                'reasoning_paths': rpaths,
                'distances': distances,
                'relation_ids': rel_ids
            }

        return path_relation_dict

    # -------------------------------------------------------------------------
    # END OF NEW CODE
    # -------------------------------------------------------------------------

    def _assembly(
        self,
        processed_dict_list,
        triple_score_dict,
        emb_dict,
        skip_no_path,
        path_relation_dict  # NEW input
    ):
        self.processed_dict_list = []

        num_relevant_triples = []
        num_skipped = 0
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['nx_graph'] = self.nx_graphs[i]

            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue
            
            # Embeddings
            # sample_i.update(emb_dict[sample_i_id]) # no need to save emb, to save memory 

            # Clean up answer entity duplicates
            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            # -------------------------------------------------------------
            # NEW: attach the path/relations info from path_relation_dict
            # -------------------------------------------------------------
            if sample_i_id in path_relation_dict:
                sample_i['translated_paths'] = path_relation_dict[sample_i_id]['translated_paths']
                sample_i['reasoning_paths'] = path_relation_dict[sample_i_id]['reasoning_paths']
                sample_i['path_distances'] = path_relation_dict[sample_i_id]['distances']
                sample_i['relation_id_tensor'] = path_relation_dict[sample_i_id]['relation_ids']
            else:
                # If not found in dict, default to empty
                sample_i['translated_paths'] = []
                sample_i['reasoning_paths'] = []
                sample_i['path_distances'] = []
                sample_i['relation_id_tensor'] = []
            # -------------------------------------------------------------

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')

    def __len__(self):
        return len(self.processed_dict_list)
    
    def __getitem__(self, i):
        return self.processed_dict_list[i]

def collate_retriever(data):
    sample = data[0]
    
    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)
    
    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)
    
    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)
    
    num_non_text_entities = len(sample['non_text_entity_list'])
    
    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'],\
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'],\
        sample['translated_paths'], sample['reasoning_paths'], sample['path_distances'], sample['relation_id_tensor'],\
        sample['id2entities'], sample['id2relations'],sample['non_text_entity_list'],sample['text_entity_list'],sample['nx_graph']


