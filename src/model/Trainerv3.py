import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SGConv, GINConv
import torch.nn.functional as F
import math

import numpy as np
import pandas as pd
import random
import sys
import os
from tqdm import tqdm
import traceback
from termcolor import colored
from src.dataset.utils import decode_path_from_indices, decode_path
import time



class Trainer(nn.Module):
    def __init__(self, model_type='GCN', bidirectional=False, use_stop_mlp=True, num_mlp_layers=2,
                 num_layers=3, num_heads=4, K=3, in_dims=3072, hidden_dims=512, out_dims=512,
                 batch_norm=True, dropout=0.5, lr=1e-3, epochs=100, num_sampled_paths=10,
                 max_depth=6, device=0, **args):
        super().__init__()
        self.run = args.get('wandb', None)
        self.last_entity_cache = []
        self.pathloss_train_after_epoch = args.get('pathloss_train_after_epoch', 0)
        self.model_type = model_type
        self.max_depth = max_depth
        self.bidirectional = bidirectional
        self.use_stop_mlp = use_stop_mlp
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.out_dims = out_dims
        self.epochs = epochs
        self.num_sampled_paths = num_sampled_paths
        self.epoch_cuda_memories = []  # To store memory in MB
        self.epoch_runtimes = []       # To store runtime in seconds
        
        # Device setup
        if isinstance(device, int):
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # MLP for question embedding
        self.q_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512)
        )

        # MLP for topic triple embedding
        self.r_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        if use_stop_mlp:
            self.stop_mlp = nn.Sequential(
                nn.Linear(1024, hidden_dims),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims, out_dims)
            )

        # ----------------------------------------------------
        # (A) If model_type is one of the GNN variants, define
        #     GNN layers exactly as before
        # ----------------------------------------------------
        if model_type in ['GCN', 'GAT', 'SGConv', 'GIN']:
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList() if batch_norm else None

            hidden = hidden_dims
            for layer in range(num_layers):
                in_dim = in_dims if layer == 0 else hidden
                out_dim = out_dims if layer == num_layers - 1 else hidden

                if model_type == 'GCN':
                    conv = GCNConv(in_dim, out_dim, add_self_loops=False)
                elif model_type == 'GAT':
                    conv = GATConv(in_dim, out_dim, heads=num_heads, add_self_loops=False)
                elif model_type == 'SGConv':
                    conv = SGConv(in_dim, out_dim, K=K, add_self_loops=False)
                elif model_type == 'GIN':
                    mlp = nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.ReLU(),
                        nn.Linear(out_dim, out_dim)
                    )
                    conv = GINConv(mlp, add_self_loops=False)
                
                self.convs.append(conv)

                if batch_norm and layer != num_layers - 1:
                    self.bns.append(nn.BatchNorm1d(out_dim))

            # If bidirectional, define the backward pass GNN layers similarly
            if bidirectional:
                self.convs_back = nn.ModuleList()
                self.bns_back = nn.ModuleList() if batch_norm else None
                for layer in range(num_layers):
                    in_dim = in_dims if layer == 0 else hidden
                    out_dim = out_dims if layer == num_layers - 1 else hidden

                    if model_type == 'GCN':
                        conv_b = GCNConv(in_dim, out_dim, add_self_loops=False)
                    elif model_type == 'GAT':
                        conv_b = GATConv(in_dim, out_dim, heads=num_heads, add_self_loops=False)
                    elif model_type == 'SGConv':
                        conv_b = SGConv(in_dim, out_dim, K=K, add_self_loops=False)
                    elif model_type == 'GIN':
                        mlp_b = nn.Sequential(
                            nn.Linear(in_dim, out_dim),
                            nn.ReLU(),
                            nn.Linear(out_dim, out_dim)
                        )
                        conv_b = GINConv(mlp_b, add_self_loops=False)
                    
                    self.convs_back.append(conv_b)

                    if batch_norm and layer != num_layers - 1:
                        self.bns_back.append(nn.BatchNorm1d(out_dim))

        # ----------------------------------------------------
        # (B) Otherwise, if model_type == "MLP", define a
        #     multi-layer MLP
        # ----------------------------------------------------
        elif model_type == 'MLP':
            # We'll store the layers in a simple Sequential
            mlp_layers = []
            hidden = hidden_dims
            current_in = in_dims
            for layer_idx in range(num_layers):
                # Decide the output dim
                current_out = out_dims if layer_idx == num_layers - 1 else hidden

                mlp_layers.append(nn.Linear(current_in, current_out))
                
                # For all but the last layer, apply ReLU + optional BN + Dropout
                if layer_idx != num_layers - 1:
                    mlp_layers.append(nn.ReLU())
                    if batch_norm:
                        mlp_layers.append(nn.BatchNorm1d(current_out))
                    mlp_layers.append(nn.Dropout(dropout))

                current_in = current_out

            self.mlp = nn.Sequential(*mlp_layers)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Stop token embedding
        self.stop_emb = nn.Parameter(torch.randn(out_dims))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Move to device
        self.to(self.device)

    def _gnn_forward(self, x, edge_index):
        # Forward direction
        h = x
        for layer in range(self.num_layers):
            h = self.convs[layer](h, edge_index)
            if layer != self.num_layers - 1:
                if self.batch_norm:
                    h = self.bns[layer](h)
                h = torch.relu(h)
        if not self.bidirectional:
            return h

        # Backward direction
        rev_edge_index = torch.flip(edge_index, [0])
        hb = x
        for layer in range(self.num_layers):
            hb = self.convs_back[layer](hb, rev_edge_index)
            if layer != self.num_layers - 1:
                if self.batch_norm:
                    hb = self.bns_back[layer](hb)
                hb = torch.relu(hb)
        return h + hb

    def train_step(self, batch, scaler, pathtrainingstart=False):
        """
        Handles a batch of graphs, with `batch` being a `DataBatch` object from PyG.

        Input:
            batch.x               -> [total_nodes, feat_dim]
            batch.edge_index      -> [2, total_edges]
            batch.one_hop_neighbors
            batch.path_label      -> [total_nodes, 6]
            batch.topic_candidates-> [total_nodes, 1]
            batch.topic_labels     -> [total_nodes, 1]  (1 indicates ground-truth topic nodes)
            batch.q_emb           -> [batch_size, q_emb_dim]
            batch.ptr             -> [batch_size+1]

        Returns:
            (total_loss.item(), topic_loss.item(), path_loss.item())
        """
        device = self.device
        batch = batch.to(device)

        total_nodes = batch.x.size(0)
        batch_size = batch.ptr.size(0) - 1

        with torch.amp.autocast(device_type='cuda'):
            # For topic-entity
            if self.model_type != "MLP":
                node_emb = self._gnn_forward(batch.x, batch.edge_index)
            else:
                node_emb = self.mlp(batch.x)
            topic_triplet_emb = self.r_mlp(batch.x[:, 1024:2048])
            topic_triplet_emb = topic_triplet_emb + node_emb

            # Process question embedding
            q_emb = batch.q_emb.to(device)
            if self.use_stop_mlp:
                cond_stop_emb = self.stop_mlp(q_emb)
            q_emb = self.q_mlp(q_emb)

            topic_loss = torch.tensor(0.0, device=device)
            path_loss = torch.tensor(0.0, device=device)
            topic_count = 0
            path_count = 0

            # -------------------
            # Process Each Graph
            # -------------------
            for graph_idx in range(batch_size):
                graph_start = batch.ptr[graph_idx].item()
                graph_end = batch.ptr[graph_idx + 1].item()
                num_nodes_in_graph = graph_end - graph_start

                # path_label has shape [num_nodes_in_graph, 6]
                path_label = batch.path_label[graph_start:graph_end]
                # topic_candidates has shape [num_nodes_in_graph,1]
                topic_candidates = batch.topic_candidates[graph_start:graph_end]
                # topic_label has shape [num_nodes_in_graph,1]
                topic_label = batch.topic_labels[graph_start:graph_end]
                one_hop_neighbors = batch.one_hop_neighbors[graph_idx]
                one_hop_neighbors_adjusted = [
                    [n + graph_start for n in neighbors] for neighbors in one_hop_neighbors
                ]

                # ------------------------------------------------------------------
                # (1) Topic-Entity Loss with Negative Sampling
                # ------------------------------------------------------------------
                # Positive set: all nodes with topic_label == 1
                pos_indices_local = (topic_label.view(-1) == 1).nonzero(as_tuple=True)[0]
                # Negative set: from topic_candidates == 1 but excluding topic_label == 1
                neg_indices_local = (
                    ((topic_candidates.view(-1) == 1) & (topic_label.view(-1) == 0))
                ).nonzero(as_tuple=True)[0]

                # Shift to batch-level indices
                pos_indices = pos_indices_local + graph_start
                neg_indices = neg_indices_local + graph_start

                # If we have no positives, skip
                if len(pos_indices) == 0:
                    # can't do topic entity training if no positive topics
                    pass
                else:
                    # sample 3 times
                    for _ in range(1):
                        # skip if not enough negatives
                        if len(neg_indices) < 3:
                            break

                        # randomly pick 1 from positives
                        pos_idx = random.choice(pos_indices.tolist())
                        # pick up to K=8 from negatives
                        K = min(8, len(neg_indices))
                        neg_sample = random.sample(neg_indices.tolist(), K)

                        # Embeddings
                        pos_emb = topic_triplet_emb[pos_idx]
                        neg_embs = topic_triplet_emb[neg_sample]

                        # Score
                        #  - pos_logit: single scalar
                        #  - neg_logit: shape [K]
                        pos_logit = torch.dot(pos_emb, q_emb[graph_idx])
                        neg_logit = torch.matmul(neg_embs, q_emb[graph_idx])  # (K,)

                        # We can do standard binary cross-entropy for positive + negative
                        # Positive label = 1, Negative label = 0
                        # => total topic loss
                        pos_loss = F.binary_cross_entropy_with_logits(
                            pos_logit.unsqueeze(0),  # (1,)
                            torch.ones(1, device=device)
                        )
                        neg_loss = F.binary_cross_entropy_with_logits(
                            neg_logit,  # (K,)
                            torch.zeros(len(neg_logit), device=device)
                        )
                        topic_loss_sample = pos_loss + neg_loss

                        topic_loss += topic_loss_sample
                        topic_count += 1

                # ------------------------------------------------------------------
                # (2) Path Loss (if pathtrainingstart == True)
                # ------------------------------------------------------------------
                # We pick exactly one valid path column in each graph (like original code).
                if pathtrainingstart:
                    # Find which columns actually have step=0
                    N_path = path_label.size(1)
                    valid_columns = []
                    for c in range(N_path):
                        column_vals = path_label[:, c]
                        if torch.any(column_vals == 0):
                            valid_columns.append(c)
                    if len(valid_columns) == 0:
                        continue

                    chosen_col = random.choice(valid_columns)
                    column_vals = path_label[:, chosen_col]
                    step_to_node = {}
                    for i_node in range(num_nodes_in_graph):
                        step = column_vals[i_node].item()
                        if step >= 0:
                            step_to_node[step] = graph_start + i_node

                    # Must have a step=0 node
                    if 0 not in step_to_node:
                        continue

                    # Ground-truth path steps
                    path_steps = sorted(step_to_node.keys())
                    current_node = step_to_node[path_steps[0]]
                    context = q_emb[graph_idx] + node_emb[current_node]

                    stop_emb = self.stop_emb.unsqueeze(0)  # shape [1, out_dims]
                    if self.use_stop_mlp:
                        stop_emb = stop_emb + cond_stop_emb[graph_idx].unsqueeze(0)

                    # If path length < 2 => immediate STOP
                    if len(path_steps) < 2:
                        neighbors = one_hop_neighbors_adjusted[current_node - graph_start]
                        cand_embs = torch.cat([node_emb[neighbors], stop_emb], dim=0)
                        logits_stop = torch.matmul(cand_embs, context)
                        target_idx_stop = torch.tensor([len(neighbors)], device=device)
                        path_loss_sample = F.cross_entropy(logits_stop.unsqueeze(0), target_idx_stop)
                        path_loss += path_loss_sample
                        path_count += 1
                    else:
                        # Autoregressive path
                        for step_i in path_steps[1:]:
                            gt_next_node = step_to_node[step_i]
                            neighbors = one_hop_neighbors_adjusted[current_node - graph_start]
                            if gt_next_node not in neighbors:
                                continue
                            # Candidate embeddings: neighbor + stop
                            cand_embs = torch.cat([node_emb[neighbors], stop_emb], dim=0)
                            logits = torch.matmul(cand_embs, context)
                            target_idx = torch.tensor([neighbors.index(gt_next_node)], device=device)
                            path_loss_sample = F.cross_entropy(logits.unsqueeze(0), target_idx)
                            path_loss += path_loss_sample
                            path_count += 1

                            # Update context
                            context = context + node_emb[gt_next_node]
                            current_node = gt_next_node

                        # Finally, STOP
                        neighbors = one_hop_neighbors_adjusted[current_node - graph_start]
                        cand_embs = torch.cat([node_emb[neighbors], stop_emb], dim=0)
                        logits_stop = torch.matmul(cand_embs, context)
                        target_idx_stop = torch.tensor([len(neighbors)], device=device)
                        path_loss_sample = F.cross_entropy(
                            logits_stop.unsqueeze(0), target_idx_stop
                        )
                        path_loss += path_loss_sample
                        path_count += 1

        # Normalize losses
        if topic_count > 0:
            topic_loss /= topic_count
        if path_count > 0:
            path_loss /= path_count
        total_loss = topic_loss + path_loss

        # Backprop
        self.optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        return total_loss.item(), topic_loss.item(), path_loss.item()

    def fit(
        self,
        dataloader1,
        val_dataloader1=None,
        save_dir=None,
        pathtrainingstart: bool = True,
    ):
        """
        Train the model on a *single* dataset and (optionally) evaluate on a single
        validation set each epoch.

        The function:
        • Performs mixed-precision training (GradScaler).
        • Logs average topic loss, path loss, and total loss per epoch.
        • Tracks the top-3 F1 scores on the validation set and checkpoints
            those models separately.
        • Saves a full model snapshot at the end of every epoch
            (   epoch_XXX_trainLoss_YYY.pt   ).

        Args
        ----
        dataloader1 : torch.utils.data.DataLoader
            Training dataloader.
        val_dataloader1 : torch.utils.data.DataLoader, optional
            Validation dataloader.
        save_dir : str, optional
            Directory for logs and checkpoints.  Created if it does not exist.
        pathtrainingstart : bool, default = False
            Whether to include path loss from epoch 0.  (If you switch this flag
            mid-training, pass the desired value here.)
        """
        scaler = torch.cuda.amp.GradScaler()

        # track top-3 validation F1
        best_f1_scores = []

        # prepare directories / offline logs
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_dir_val = os.path.join(save_dir, "val_set")
            os.makedirs(save_dir_val, exist_ok=True)
            log_file = os.path.join(save_dir, "oom_log.txt")
            log_arrays = {
                'TrainTopicLoss': [], 'TrainPathLoss': [], 'TrainTotalLoss': [],
                'ValTopicLoss':   [], 'ValPathLoss':   [],
                'Precision':      [], 'Recall':        [], 'F1': []
            }
        else:
            log_file = "oom_log.txt"
            log_arrays = None

        for epoch in range(1, self.epochs + 1):
            # enable path loss immediately (set externally if you need delays)
            pathtrainingstart = True

            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            self.train()

            # accumulators
            tot_loss = tot_topic = tot_path = 0.0
            batch_cnt = 0

            ################################################
            # ----------  Training loop  -------------------
            ################################################
            for batch_idx, batch in enumerate(dataloader1):
                try:
                    loss, topic_loss, path_loss = self.train_step(
                        batch, scaler, pathtrainingstart
                    )
                    tot_loss  += loss
                    tot_topic += topic_loss
                    tot_path  += path_loss
                    batch_cnt += 1

                    print(f"[Epoch {epoch} | Batch {batch_cnt}] "
                        f"TopicLoss: {topic_loss:.4f}, "
                        f"PathLoss: {path_loss:.4f}, "
                        f"TotalLoss: {loss:.4f}")

                except RuntimeError as e:
                    print(f"RuntimeError at batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    if "memory" in str(e).lower():
                        with open(log_file, "a") as f:
                            f.write(f"[OOM] Epoch {epoch}, Batch {batch_idx}\n"
                                    f"{traceback.format_exc()}\n"
                                    "-------------------------\n")
                    continue  # skip the failed batch

            # epoch-level statistics
            avg_loss       = tot_loss  / max(1, batch_cnt)
            avg_topic_loss = tot_topic / max(1, batch_cnt)
            avg_path_loss  = tot_path  / max(1, batch_cnt)

            print(f"Epoch {epoch} summary — "
                f"AvgLoss: {avg_loss:.4f}, "
                f"AvgTopicLoss: {avg_topic_loss:.4f}, "
                f"AvgPathLoss: {avg_path_loss:.4f} ")

            ################################################
            # ----------  Offline / wandb logging ----------
            ################################################
            if self.run is not None:            # wandb
                self.run.log({
                    'TrainTopicLoss': avg_topic_loss,
                    'TrainPathLoss':  avg_path_loss,
                    'TrainTotalLoss': avg_loss,
                })
            elif log_arrays is not None:        # offline
                log_arrays['TrainTopicLoss'].append(avg_topic_loss)
                log_arrays['TrainPathLoss'].append(avg_path_loss)
                log_arrays['TrainTotalLoss'].append(avg_loss)

            ################################################
            # ----------  Validation (optional) ------------
            ################################################
            if val_dataloader1 is not None and epoch > 5:
                val_topic, val_path, prec, rec = self.evaluate(
                    val_dataloader1,
                    eval_loss=True,
                    eval_pr=pathtrainingstart,
                    is_valid_or_test=True,
                    pathtrainingstart=pathtrainingstart,
                    dataset_idx=1
                )

                if pathtrainingstart:
                    print(f"[Validation] TopicLoss: {val_topic:.4f}, "
                        f"PathLoss: {val_path:.4f}, "
                        f"Precision: {prec:.4f}, Recall: {rec:.4f}")
                else:
                    print(f"[Validation] TopicLoss: {val_topic:.4f}")

                # compute F1 (only if precision/recall valid)
                f1 = 0.0
                if pathtrainingstart and prec is not None and rec is not None:
                    if prec + rec > 0:
                        f1 = 2 * prec * rec / (prec + rec)
                    print(f"[Validation] F1: {f1:.4f}")

                # wandb / offline log
                if self.run is not None:
                    payload = {'ValTopicLoss': val_topic}
                    if pathtrainingstart:
                        payload.update({
                            'ValPathLoss': val_path,
                            'Precision':   prec,
                            'Recall':      rec,
                            'F1':          f1,
                        })
                    self.run.log(payload)
                elif log_arrays is not None:
                    log_arrays['ValTopicLoss'].append(val_topic)
                    if pathtrainingstart:
                        log_arrays['ValPathLoss'].append(val_path)
                        log_arrays['Precision'].append(prec)
                        log_arrays['Recall'].append(rec)
                        log_arrays['F1'].append(f1)

                # checkpoint if within top-3 F1
                if pathtrainingstart and f1 > 0:
                    if len(best_f1_scores) < 3 or f1 > min(best_f1_scores):
                        if len(best_f1_scores) == 3:
                            best_f1_scores.remove(min(best_f1_scores))
                        best_f1_scores.append(f1)
                        best_f1_scores.sort(reverse=True)

                        if save_dir is not None:
                            ckpt_path = os.path.join(
                                save_dir_val,
                                f"model_epoch_{epoch}_f1_{f1:.4f}.pt"
                            )
                            torch.save(self.state_dict(), ckpt_path)
                            print(f"[Validation] Checkpoint saved → {ckpt_path}")

            ################################################
            # ----------  Flush offline logs ---------------
            ################################################
            if self.run is None and log_arrays is not None and save_dir is not None:
                for k, v in log_arrays.items():
                    if not v:
                        continue
                    np.save(os.path.join(save_dir, f"{k}.npy"),
                            np.array([x.cpu().numpy() if torch.is_tensor(x) else x
                                    for x in v]))
                print(f"Saved offline logs → {save_dir}")

            ################################################
            # ----------  Epoch-level checkpoint -----------
            ################################################
            if save_dir is not None:
                ckpt_name = f"epoch_{epoch}_trainLoss_{avg_loss:.4f}.pt"
                torch.save(self.state_dict(), os.path.join(save_dir, ckpt_name))
                print(f"[Epoch {epoch}] Full model saved → {ckpt_name}")

        print("Training complete.")
        print("Peak memory (first 5 epochs):", self.epoch_cuda_memories[:5])
        print("Runtime     (first 5 epochs):", self.epoch_runtimes[:5])





    def evaluate(self, 
                dataloader, 
                eval_loss=True, 
                eval_pr=True, 
                is_valid_or_test=True, 
                pathtrainingstart=False, 
                dataset_idx=0,K=50,N=1,M=50):
        """
        Evaluates the model on a given dataloader, assuming each batch has batch_size=1.
        Also accumulates the total number of nodes used in the sampled paths.
        """
        print("Evaluating the model...")
        self.eval()
        device = self.device

        total_topic_loss = 0.0
        total_path_loss = 0.0

        # For recall
        total_correct = 0
        total_true = 0
        total_pred = 0  # if you also want to track precision

        # NEW: track total number of nodes in sampled paths
        total_sampled_nodes = 0
        total_hit = 0.

        # Ensure self.last_entity_cache exists up to the specified index
        if len(self.last_entity_cache) <= dataset_idx:
            for _ in range(dataset_idx - len(self.last_entity_cache) + 1):
                self.last_entity_cache.append({})

        cache_dict = self.last_entity_cache[dataset_idx]

        global_graph_idx = 0
        N_batch = len(dataloader)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                for batch_idx, batch in enumerate(tqdm(dataloader)):
                    # Because batch_size=1, 'batch' is a single graph
                    batch = batch.to(device)

                    # Evaluate node embeddings (for topic loss)
                    node_emb = self._gnn_forward(batch.x, batch.edge_index)
                    topic_triplet_emb = self.r_mlp(batch.x[:, 1024:2048]) + node_emb

                    # Process question embedding
                    q_ = batch.q_emb.to(device)
                    if self.use_stop_mlp:
                        cond_stop_emb = self.stop_mlp(q_)
                    q_ = self.q_mlp(q_)

                    # Compute losses if requested
                    if eval_loss:
                        batch_topic_loss = 0.0
                        batch_path_loss = 0.0
                        topic_count = 0
                        path_count = 0

                        # (1) Topic Loss
                        topic_candidates = batch.topic_candidates.view(-1)
                        topic_labels = batch.topic_labels.view(-1)
                        pos_indices_local = (topic_labels == 1).nonzero(as_tuple=True)[0]
                        neg_indices_local = ((topic_candidates == 1) & (topic_labels == 0)).nonzero(as_tuple=True)[0]

                        if len(pos_indices_local) > 0:
                            for _ in range(2):
                                if len(neg_indices_local) < 3:
                                    break
                                pos_idx = random.choice(pos_indices_local.tolist())
                                K_ = min(8, len(neg_indices_local))
                                neg_sample = random.sample(neg_indices_local.tolist(), K_)

                                pos_emb = topic_triplet_emb[pos_idx]
                                neg_embs = topic_triplet_emb[neg_sample]

                                pos_logit = torch.dot(pos_emb, q_[0])
                                neg_logit = torch.matmul(neg_embs, q_[0])

                                pos_loss = F.binary_cross_entropy_with_logits(
                                    pos_logit.unsqueeze(0),
                                    torch.ones(1, device=device)
                                )
                                neg_loss = F.binary_cross_entropy_with_logits(
                                    neg_logit,
                                    torch.zeros(len(neg_logit), device=device)
                                )
                                topic_loss_sample = pos_loss + neg_loss
                                batch_topic_loss += topic_loss_sample
                                topic_count += 1

                        # (2) Path Loss
                        if pathtrainingstart:
                            path_label = batch.path_label
                            node_count = path_label.size(0)
                            valid_columns = [
                                c for c in range(path_label.size(1))
                                if torch.any(path_label[:, c] == 0)
                            ]
                            for ccol in valid_columns:
                                col_vals = path_label[:, ccol]
                                stn = {
                                    col_vals[i].item(): i
                                    for i in range(node_count) if col_vals[i] >= 0
                                }
                                if 0 not in stn:
                                    continue
                                path_steps = sorted(stn.keys())
                                current_node = stn[path_steps[0]]
                                context = q_[0] + node_emb[current_node]

                                stop_emb_ = self.stop_emb.unsqueeze(0)
                                if self.use_stop_mlp:
                                    stop_emb_ = stop_emb_ + cond_stop_emb[0].unsqueeze(0)

                                if len(path_steps) < 2:
                                    neighbors = batch.one_hop_neighbors[current_node]
                                    c_embs = torch.cat([node_emb[neighbors], stop_emb_], dim=0)
                                    logits_stop = torch.matmul(c_embs, context)
                                    target_idx_stop = torch.tensor([len(neighbors)], device=device)
                                    path_loss_sample = F.cross_entropy(
                                        logits_stop.unsqueeze(0), target_idx_stop
                                    )
                                    batch_path_loss += path_loss_sample
                                    path_count += 1
                                else:
                                    for step_i in path_steps[1:]:
                                        gt_next_node = stn[step_i]
                                        neighbors = batch.one_hop_neighbors[current_node]
                                        if gt_next_node not in neighbors:
                                            continue
                                        c_embs = torch.cat([node_emb[neighbors], stop_emb_], dim=0)
                                        logits_ = torch.matmul(c_embs, context)
                                        target_idx_ = torch.tensor([neighbors.index(gt_next_node)], device=device)
                                        path_loss_sample = F.cross_entropy(
                                            logits_.unsqueeze(0), target_idx_
                                        )
                                        batch_path_loss += path_loss_sample
                                        path_count += 1
                                        context = context + node_emb[gt_next_node]
                                        current_node = gt_next_node

                                    # final STOP
                                    neighbors = batch.one_hop_neighbors[current_node]
                                    c_embs = torch.cat([node_emb[neighbors], stop_emb_], dim=0)
                                    logits_stop = torch.matmul(c_embs, context)
                                    target_idx_stop = torch.tensor([len(neighbors)], device=device)
                                    path_loss_sample = F.cross_entropy(
                                        logits_stop.unsqueeze(0), target_idx_stop
                                    )
                                    batch_path_loss += path_loss_sample
                                    path_count += 1

                        if topic_count > 0:
                            batch_topic_loss /= topic_count
                        if path_count > 0:
                            batch_path_loss /= path_count
                        total_topic_loss += batch_topic_loss
                        total_path_loss += batch_path_loss

                    # (B) Evaluate Recall + track # of nodes in sampled paths
                    if 0:
                        # call decoding
                        top_paths = self.decoding(
                            batch,
                            retrieval_dict=None, 
                            metadata_list=None,
                            K=K,
                            N=N,
                            M=M,
                            way='normal'
                        )
                        # top_paths[0] is a list of (path_idxs, prob)

                        # accumulate the total number of sampled nodes
                        # from each path
                        for (path_idxs, path_prob) in top_paths[0]:
                            total_sampled_nodes += len(path_idxs)

                        # gather predicted nodes (union)
                        pred_nodes = set()
                        for (path_idxs, path_prob) in top_paths[0]:
                            pred_nodes.update(path_idxs)

                        # gather ground-truth nodes from path_label 
                        path_label = batch.path_label
                        node_count, col_count = path_label.shape
                        gt_nodes = set()
                        for col_i in range(col_count):
                            col_vals = path_label[:, col_i]
                            if torch.all(col_vals == -1):
                                continue
                            max_step = col_vals.max().item()
                            if max_step < 0:
                                continue
                            max_idxs = (col_vals == max_step).nonzero(as_tuple=True)[0]
                            for x_ in max_idxs:
                                gt_nodes.add(x_.item())

                        if len(gt_nodes) > 0:
                            n_covered = len(pred_nodes & gt_nodes)
                            total_correct += n_covered
                            if n_covered > 0:
                                total_hit += 1
                            total_true += len(gt_nodes)
                            total_pred += len(pred_nodes)

                    global_graph_idx += 1

        # compute average losses
        avg_topic_loss = total_topic_loss / N_batch if eval_loss else None
        avg_path_loss = total_path_loss / N_batch if eval_loss else None

        print(f"Avg Topic Loss: {avg_topic_loss}, Avg Path Loss: {avg_path_loss}")
        print(f"Avg number of nodes across all sampled paths in this dataset: {total_sampled_nodes/global_graph_idx}")

        return avg_topic_loss, avg_path_loss, None, None, total_sampled_nodes, total_hit/global_graph_idx


    def cond_evaluate(self, dataloader, eval_pr=True, is_valid_or_test=True,
                      pathtrainingstart=False, metadata_list=None, id_entity_mapping_list=None):
        """
        A specialized evaluation method that decodes paths (for debugging or logging).
        """
        print("Evaluating the model with path decoding (cond_evaluate)...")
        self.eval()
        device = self.device

        total_correct = 0
        total_pred = 0
        total_true = 0

        wrong_indices = []
        wrong_paths = []

        last_entity_cache = {}
        global_graph_idx = 0
        N_batch = len(dataloader)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                for batch in tqdm(dataloader):
                    if metadata_list is not None:
                        # For demonstration: get ground truth path from decoding
                        gt_path = decode_path(batch,
                                              retrieval_dict=id_entity_mapping_list[global_graph_idx],
                                              metadata_list=metadata_list[global_graph_idx])

                    batch = batch.to(device)
                    batch_size = batch.ptr.size(0) - 1

                    topic_triplet_emb = self.r_mlp(batch.x[:, 1024:2048])
                    if pathtrainingstart:
                        node_emb = self._gnn_forward(batch.x, batch.edge_index)
                    q_emb = batch.q_emb.to(device)
                    if self.use_stop_mlp:
                        cond_stop_emb = self.stop_mlp(q_emb)
                    q_emb = self.q_mlp(q_emb)

                    for graph_idx in range(batch_size):
                        graph_start = batch.ptr[graph_idx].item()
                        graph_end = batch.ptr[graph_idx + 1].item()
                        num_nodes_in_graph = graph_end - graph_start

                        path_label = batch.path_label[graph_start:graph_end]
                        topic_candidates = batch.topic_candidates[graph_start:graph_end]
                        topic_label = batch.topic_labels[graph_start:graph_end]
                        one_hop_neighbors = batch.one_hop_neighbors[graph_idx]

                        one_hop_neighbors_adjusted = [
                            [n + graph_start for n in neighbors] for neighbors in one_hop_neighbors
                        ]

                        valid_columns = [
                            c for c in range(path_label.size(1))
                            if torch.any(path_label[:, c] == 0)
                        ]
                        if len(valid_columns) == 0:
                            global_graph_idx += 1
                            continue

                        if eval_pr and pathtrainingstart:
                            global_graph_id = global_graph_idx
                            if is_valid_or_test and global_graph_id in last_entity_cache:
                                gt_last_entities = last_entity_cache[global_graph_id]
                            else:
                                gt_last_entities = set()
                                for col in valid_columns:
                                    col_vals = path_label[:, col]
                                    stn = {
                                        col_vals[i].item(): graph_start + i
                                        for i in range(num_nodes_in_graph) if col_vals[i] >= 0
                                    }
                                    if stn:
                                        last_step = max(stn.keys())
                                        gt_last_entities.add(stn[last_step])
                                if is_valid_or_test:
                                    last_entity_cache[global_graph_id] = gt_last_entities

                            # Gather topic nodes
                            topic_label_local = (topic_label.view(-1) == 1).nonzero(as_tuple=True)[0]
                            topic_label_nodes = set((topic_label_local + graph_start).tolist())

                            # From topic_candidates, sample 5 more
                            candidate_local_indices = (topic_candidates.view(-1) == 1).nonzero(as_tuple=True)[0]
                            candidate_batch_indices = candidate_local_indices + graph_start
                            topic_candidate_nodes = set(topic_label_nodes)

                            if len(candidate_batch_indices) > 0:
                                cand_embs_ = node_emb[candidate_batch_indices]
                                logits_ = torch.matmul(cand_embs_, q_emb[graph_idx])
                                probs_ = F.softmax(logits_, dim=0)
                                for _ in range(5):
                                    smp_idx = torch.multinomial(probs_, 1).item()
                                    node_added = candidate_batch_indices[smp_idx].item()
                                    topic_candidate_nodes.add(node_added)

                            sampled_entities = set()
                            sampled_paths = []
                            for t_node in topic_candidate_nodes:
                                for _ in range(self.num_sampled_paths):
                                    current_node = t_node
                                    path_ = [current_node]
                                    context = q_emb[graph_idx] + node_emb[current_node]

                                    for __ in range(self.max_depth):
                                        neighbors = one_hop_neighbors_adjusted[current_node - graph_start]
                                        stop_emb_ = self.stop_emb.unsqueeze(0)
                                        if self.use_stop_mlp:
                                            stop_emb_ = stop_emb_ + cond_stop_emb[graph_idx].unsqueeze(0)

                                        c_embs = torch.cat([node_emb[neighbors], stop_emb_], dim=0)
                                        logits_step = torch.matmul(c_embs, context)
                                        probs_step = F.softmax(logits_step, dim=0)
                                        sampled_idx = torch.multinomial(probs_step, 1).item()

                                        if sampled_idx == len(neighbors):
                                            break
                                        current_node = neighbors[sampled_idx]
                                        path_.append(current_node)
                                        context = context + node_emb[current_node]

                                    sampled_entities.add(current_node)
                                    sampled_paths.append(path_)

                            if len(gt_last_entities) > 0:
                                true_positive = len(sampled_entities & gt_last_entities)
                                total_correct += true_positive
                                total_pred += len(sampled_entities)
                                total_true += len(gt_last_entities)

                            # Optionally log wrong paths
                            if true_positive == 0 and metadata_list is not None:
                                text_paths = decode_path_from_indices(
                                    sampled_paths,
                                    id_entity_mapping_list[global_graph_idx],
                                    metadata_list[global_graph_idx]
                                )
                                os.makedirs("experiments/wrong_paths", exist_ok=True)
                                file_path = os.path.join(f"experiments/wrong_paths/graph_{global_graph_idx}_paths.txt")
                                with open(file_path, "w") as f:
                                    f.write("Sampled Paths:\n")
                                    for pth in text_paths:
                                        f.write(f"{pth}\n")
                                    f.write("\nGround Truth Path:\n")
                                    for pth in gt_path:
                                        f.write(f"{pth}\n")

                        global_graph_idx += 1

        precision = None
        recall = None
        if eval_pr and total_pred > 0 and total_true > 0:
            precision = total_correct / total_pred
            recall = total_correct / total_true
        elif eval_pr:
            if total_true > 0:
                recall = 0.0

        print(f"Total precision: {precision}, recall: {recall}")
        return precision, recall


    def decoding(
        self,
        batch,
        retrieval_dict,
        metadata_list,
        K=50,
        N=2,
        M=50,           
        way='greedy'  
    ):
        """
        Decodes paths from a single PyG Data object (no batching).
        Uses 'topic_triplet_emb' only for topic-entity selection scores, 
        and uses 'node_emb' during path sampling expansions.

        Parameters
        ----------
        batch : Data
            A PyG Data object containing:
                - x : node features
                - edge_index : edge indices
                - topic_labels : [num_nodes] or [num_nodes, 1]
                - topic_candidates : [num_nodes] or [num_nodes, 1]
                - q_emb : [1, q_dim] for question embedding
                - one_hop_neighbors : list of lists of neighbor indices
                - max_depth : (optional) if not set, we use self.max_depth
                (plus any other attributes needed).

        retrieval_dict : dict
            Contains 'id2entities' and 'id2relations' for textual decoding.

        metadata_list : list of dict
            Each entry corresponds to a node index:
                { 'relation_id': r_id, 'h_id': h_idx, 't_id': t_idx }
            Used by decode_path_from_indices(...) for textual mapping.

        K : int
            Number of sampled topic nodes from candidate distribution.

        N : int
            Number of expansions from each topic node.

        M : int
            Number of top paths to keep (sorted by probability) after expansions.

        way : str
            Decoding strategy. Either "greedy" or "normal".

        Returns
        -------
        results : list
            A list with a single element (since there's only one graph),
            that element is a list of up to M decoded paths. 
            Each path is a list of textual triples: (head_text, relation_text, tail_text).
        """
        
        self.eval()
        device = self.device

        batch = batch.to(device)
        # We only handle a single-graph Data object
        # So there's no batch loop.

        # Forward pass to compute embeddings
        with torch.no_grad():
            # (A) GNN node embeddings (for path expansions)
            node_emb = self._gnn_forward(batch.x, batch.edge_index)
            # (B) 'topic_triplet_emb' for selecting topic entities
            topic_triplet_emb = self.r_mlp(batch.x[:, 1024:2048]) + node_emb

            # We expect q_emb to be [1, q_dim]. We'll just use q_emb_single = q_emb[0].
            q_emb = batch.q_emb.float().to(device)          # shape: [1, q_dim]
            
            cond_stop_emb = None
            if self.use_stop_mlp:
                cond_stop_emb = self.stop_mlp(q_emb)  # shape: [1, out_dims]
                cond_stop_emb = cond_stop_emb[0]      # shape: [out_dims]

            q_emb = self.q_mlp(q_emb)
            q_emb_single = q_emb[0]                # shape: [q_dim]
        # Prepare some utilities
        max_depth = self.max_depth  # either from batch or from self

        def compute_path_prob(step_probs):
            # step_probs is a list of floats
            log_p = sum(math.log(max(1e-15, p)) for p in step_probs)
            return math.exp(log_p)

        graph_start = 0
        graph_end = batch.x.size(0)

        topic_label = batch.topic_labels[graph_start:graph_end].view(-1)
        topic_candidates = batch.topic_candidates[graph_start:graph_end].view(-1)

        # For neighbor list
        if "ptr" in batch:
            one_hop_neighbors = batch.one_hop_neighbors[0]  # a list of lists
        else:
            one_hop_neighbors = batch.one_hop_neighbors
        one_hop_neighbors_adjusted = [
            [n + graph_start for n in neighbors] for neighbors in one_hop_neighbors
        ]
        # Label-based topic nodes
        topic_label_local = (topic_label == 1).nonzero(as_tuple=True)[0]
        topic_label_nodes = set((topic_label_local + graph_start).tolist())

        # Candidate nodes
        candidate_local_indices = (topic_candidates == 1).nonzero(as_tuple=True)[0]
        candidate_batch_indices = candidate_local_indices + graph_start

        # Start with label-based topics
        topic_candidate_nodes = set(topic_label_nodes)

        # (1) Sample K candidates from topic_triplet_emb vs q_emb_single
        if len(candidate_batch_indices) > 0:
            cand_embs_ = topic_triplet_emb[candidate_batch_indices]  # shape: [num_candidates, emb_dim]
            logits_ = torch.matmul(cand_embs_, q_emb_single)         # shape: [num_candidates]
            probs_ = F.softmax(logits_, dim=0)
            try:
                top_k_idx = torch.multinomial(probs_, K, replacement=True)  # shape: [K]
            except:
                top_k_idx = torch.multinomial(probs_, K,replacement=True)
            if way == 'greedy':
                top_k_nodes = candidate_batch_indices[top_k_idx].tolist()
                topic_candidate_nodes = set(top_k_nodes)
            else:
                # normal => unify top-K with label-based nodes
                for idx_ in top_k_idx:
                    node_ = candidate_batch_indices[idx_].item()
                    topic_candidate_nodes.add(node_)

        # We'll store (decoded_path, path_prob) in path_candidates
        path_candidates = []

        def decode_and_store(path_indices, path_prob):
            
            if retrieval_dict is not None and metadata_list is not None:
                list_of_paths = decode_path_from_indices([path_indices], retrieval_dict, metadata_list)
                decoded_path = list_of_paths[0]  # single path
                path_candidates.append((decoded_path, path_prob))
            else:
                path_candidates.append((path_indices, path_prob))

        # Greedy expansion
        def expand_path_greedy(start_node):
            """
            Expand from start_node greedily, picking argmax neighbor at each step.
            Returns exactly 1 path (plus probability).
            """
            context = q_emb_single + node_emb[start_node]
            path_ = [start_node]
            step_probs = []

            for _step in range(max_depth):
                neighbors = one_hop_neighbors_adjusted[path_[-1] - graph_start]

                # STOP embedding
                stop_emb_ = self.stop_emb.unsqueeze(0)  # shape: [1, out_dims]
                if self.use_stop_mlp and (cond_stop_emb is not None):
                    stop_emb_ = stop_emb_ + cond_stop_emb.unsqueeze(0)

                c_embs = torch.cat([node_emb[neighbors], stop_emb_], dim=0)
                logits_step = torch.matmul(c_embs, context)
                probs_step = F.softmax(logits_step, dim=0)

                argmax_idx = torch.argmax(probs_step).item()
                chosen_prob = probs_step[argmax_idx].item()
                step_probs.append(chosen_prob)

                # STOP?
                if argmax_idx == len(neighbors):
                    break

                next_node = neighbors[argmax_idx]
                path_.append(next_node)
                context = context + node_emb[next_node]

            path_prob = compute_path_prob(step_probs)
            return path_, path_prob

        # Random expansions
        def expand_path_random(start_node):
            expansions = []
            for _ in range(N):
                context = q_emb_single + node_emb[start_node]
                path_ = [start_node]
                step_probs = []

                for __ in range(max_depth):
                    neighbors = one_hop_neighbors_adjusted[path_[-1] - graph_start]

                    stop_emb_ = self.stop_emb.unsqueeze(0)
                    if self.use_stop_mlp and (cond_stop_emb is not None):
                        stop_emb_ = stop_emb_ + cond_stop_emb.unsqueeze(0)

                    c_embs = torch.cat([node_emb[neighbors], stop_emb_], dim=0)
                    logits_step = torch.matmul(c_embs, context)
                    probs_step = F.softmax(logits_step, dim=0)

                    sampled_idx = torch.multinomial(probs_step, 1).item()
                    chosen_prob = probs_step[sampled_idx].item()
                    step_probs.append(chosen_prob)

                    # STOP chosen
                    if sampled_idx == len(neighbors):
                        break

                    next_node = neighbors[sampled_idx]
                    path_.append(next_node)
                    context = context + node_emb[next_node]

                expansions.append((path_, compute_path_prob(step_probs)))
            return expansions

        # Expand from each topic node
        for t_node in topic_candidate_nodes:
            if way == 'greedy':
                # Typically you'd set N=1 for strict greedy,
                # or do multiple expansions if you want tie-breaking, etc.
                for _ in range(N):
                    pth, pprob = expand_path_greedy(t_node)
                    
                    decode_and_store(pth, pprob)
            else:
                expansions = expand_path_random(t_node)
                for (pth, pprob) in expansions:
                    decode_and_store(pth, pprob)

        # Keep top M by probability
        path_candidates.sort(key=lambda x: x[1], reverse=True)
        top_paths = path_candidates[:M]
        top_paths = [i for i in top_paths if i[1]>=1e-4]
        
        # We'll wrap it in a list for consistency
        results = [top_paths]
        return results
    
    