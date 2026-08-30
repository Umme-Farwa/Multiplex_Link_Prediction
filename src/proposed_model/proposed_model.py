# ==========================================
# IMPORT PATH FIX
# ==========================================

import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.append(PROJECT_ROOT)

# ==========================================
# IMPORTS
# ==========================================

# ==========================================
# IMPORT PATH FIX
# ==========================================

import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.append(PROJECT_ROOT)

# ==========================================
# IMPORTS
# ==========================================

import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from src.evaluation.metrics import evaluate_all, find_best_threshold
from src.utils.save_results import (
    save_metrics,
    save_predictions
)


# ==========================================
# Weak-Tie Score (Granovetter-style)
# ==========================================
#
# For each edge (u, v), weak-tie strength is estimated using the
# neighborhood overlap (topological embeddedness) between u and v.
# Low overlap -> the edge is a structural "bridge" / weak tie
# (Granovetter, 1973). We convert this into a score in [0, 1]
# where HIGH score = WEAK tie, so the model can learn to pay
# attention to these bridging edges.

def compute_weak_tie_scores(graph):

    scores = {}

    for u, v in graph.edges():

        neighbors_u = set(graph.neighbors(u)) - {v}
        neighbors_v = set(graph.neighbors(v)) - {u}

        union = neighbors_u | neighbors_v

        if len(union) == 0:
            jaccard = 0.0
        else:
            jaccard = len(neighbors_u & neighbors_v) / len(union)

        weak_tie_score = 1.0 - jaccard

        scores[(u, v)] = weak_tie_score
        scores[(v, u)] = weak_tie_score

    return scores


# ==========================================
# Create Training Graph (uses GLOBAL node set)
# ==========================================
#
# IMPORTANT: unlike the single-layer baselines, every layer graph here
# is built over the SAME global node set (union of nodes across all
# layers). This guarantees that node index i refers to the exact same
# entity in every layer, which is required for cross-layer attention
# to combine embeddings meaningfully.

def create_training_graph(global_nodes, train_edges):
    train_graph = nx.Graph()
    train_graph.add_nodes_from(global_nodes)
    train_graph.add_edges_from(train_edges)
    return train_graph


# ==========================================
# Structural Node Features
# ==========================================

def create_node_features(graph):

    nodes = sorted(graph.nodes())
    num_nodes = len(nodes)

    degree_centrality = nx.degree_centrality(graph)
    pagerank = nx.pagerank(graph)
    clustering = nx.clustering(graph)

    try:
        kcore = nx.core_number(graph)
    except Exception:
        kcore = {node: 0 for node in nodes}

    # Betweenness centrality: how often a node sits on shortest paths
    # between other nodes -- captures "bridging" structural role,
    # complementary to the local weak-tie edge score. Approximated
    # via sampling (k pivots) for speed on larger layers.
    try:
        k_pivots = min(500, num_nodes)
        betweenness = nx.betweenness_centrality(
            graph,
            k=k_pivots,
            seed=42
        )
    except Exception:
        betweenness = {node: 0.0 for node in nodes}

    # Eigenvector centrality: importance based on being connected to
    # other important nodes -- adds a "global influence" signal that
    # degree/pagerank alone don't fully capture.
    try:
        eigenvector = nx.eigenvector_centrality(
            graph,
            max_iter=1000,
            tol=1e-04
        )
    except Exception:
        eigenvector = {node: 0.0 for node in nodes}

    features = []
    for node in nodes:
        features.append([
            degree_centrality[node],
            pagerank[node],
            clustering[node],
            kcore[node],
            betweenness[node],
            eigenvector.get(node, 0.0)
        ])

    x = torch.tensor(features, dtype=torch.float)

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    x = (x - mean) / (std + 1e-8)

    return x


# ==========================================
# Convert Graph to PyG Data (with weak-tie edge_attr)
# ==========================================

def convert_to_pyg(graph, global_node_mapping):

    weak_tie_scores = compute_weak_tie_scores(graph)

    edge_list = []
    edge_attr_list = []

    for u, v in graph.edges():
        edge_list.append([global_node_mapping[u], global_node_mapping[v]])
        edge_attr_list.append(weak_tie_scores[(u, v)])

    if len(edge_list) > 0:

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).view(-1, 1)

        # symmetrize edges AND keep edge_attr aligned
        edge_index, edge_attr = to_undirected(
            edge_index,
            edge_attr,
            reduce="mean"
        )

    else:

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    x = create_node_features(graph)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


# ==========================================
# Weak-Tie-Aware GAT Encoder (per layer)
# ==========================================
#
# Same two-layer GAT design as the GAT baseline, but edge_attr
# (the weak-tie score) is injected into the attention computation
# via GATConv's edge_dim argument. This lets the attention
# coefficients depend not just on node content, but also on how
# "weak"/bridging the connecting edge is.

class WeakTieGATEncoder(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim=64,
            output_dim=32,
            heads=8,
            dropout=0.5
    ):

        super().__init__()

        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
            concat=True
        )

        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=output_dim,
            heads=1,
            edge_dim=1,
            concat=False,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr):

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)

        return x


# ==========================================
# Cross-Layer Semantic Attention
# ==========================================
#
# Learns ONE global importance weight per layer (not per node),
# following the semantic-level attention idea from HAN
# (Wang et al., 2019). This directly implements the "attention
# mechanism to identify the importance of different network
# layers" described in the proposal, and gives an interpretable
# beta score per layer that can be reported in the thesis.

class LayerAttention(nn.Module):

    def __init__(self, embedding_dim, hidden_dim=128):

        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, layer_embeddings):

        # layer_embeddings: list of [num_nodes, dim] tensors, one per layer
        Z = torch.stack(layer_embeddings, dim=0)  # [L, N, D]

        # score each layer's overall content, then average over nodes
        # to get a single importance logit per layer
        scores = self.project(Z).mean(dim=1)  # [L, 1]

        beta = torch.softmax(scores, dim=0)  # [L, 1]  layer importance weights

        beta_expanded = beta.unsqueeze(-1)  # [L, 1, 1]

        z_fused = (beta_expanded * Z).sum(dim=0)  # [N, D]

        return z_fused, beta.squeeze(-1)


# ==========================================
# Fusion Gate (combine per-layer + fused embedding)
# ==========================================
#
# For predicting a link WITHIN a specific layer, we don't want to
# throw away that layer's own structure -- so we combine the
# layer-specific embedding with the globally-fused cross-layer
# embedding. This is what allows sparse layers to borrow signal
# from richer layers for the SAME node.

class FusionGate(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, z_local, z_fused):

        combined = torch.cat([z_local, z_fused], dim=-1)

        gated = self.mlp(combined)

        # residual connection: keep the layer's own local signal
        # intact even if the fusion pathway learns something noisy
        out = z_local + gated

        return self.norm(out)


# ==========================================
# MLP Edge Decoder (vectorized, shared across layers)
# ==========================================

class EdgeDecoder(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, embeddings, edges):
        if not torch.is_tensor(edges):
            edges = torch.tensor(edges, dtype=torch.long, device=embeddings.device)

        u_emb = embeddings[edges[:, 0]]
        v_emb = embeddings[edges[:, 1]]

        pairs = torch.cat([u_emb, v_emb], dim=1)

        return self.decoder(pairs).squeeze(-1)


# ==========================================
# Full Proposed Model
# ==========================================

class ProposedMultiplexModel(nn.Module):

    def __init__(
            self,
            layer_names,
            input_dim,
            hidden_dim=64,
            embedding_dim=32,
            heads=8,
            dropout=0.5
    ):

        super().__init__()

        self.layer_names = layer_names

        # nn.ModuleDict does not allow "." in keys (used internally as a
        # submodule path separator), but layer names like "physics.soc-ph"
        # contain dots. So we map each layer name to a safe key here,
        # while everywhere else in the code we keep using the original
        # layer_name (for data/results lookups).
        self.safe_keys = {
            layer_name: layer_name.replace(".", "__DOT__")
            for layer_name in layer_names
        }

        # one weak-tie-aware GAT encoder PER layer
        self.encoders = nn.ModuleDict({
            self.safe_keys[layer_name]: WeakTieGATEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                heads=heads,
                dropout=dropout
            )
            for layer_name in layer_names
        })

        self.layer_attention = LayerAttention(embedding_dim)

        self.fusion_gate = FusionGate(embedding_dim)

        self.decoder = EdgeDecoder(embedding_dim)

    def encode(self, layer_data):

        # layer_data: dict {layer_name: PyG Data}, all sharing the
        # same global node indexing

        layer_embeddings = []

        for layer_name in self.layer_names:

            data = layer_data[layer_name]

            z = self.encoders[self.safe_keys[layer_name]](
                data.x,
                data.edge_index,
                data.edge_attr
            )

            layer_embeddings.append(z)

        z_fused, beta = self.layer_attention(layer_embeddings)

        # final per-layer representation = local structure + global
        # cross-layer context, gated together
        final_embeddings = {}

        for layer_name, z_local in zip(self.layer_names, layer_embeddings):
            final_embeddings[layer_name] = self.fusion_gate(z_local, z_fused)

        return final_embeddings, beta

    def decode(self, embeddings, layer_name, edges):
        return self.decoder(embeddings[layer_name], edges)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # --- multi-seed support (backward compatible) ---
    # THESIS_SEED     : which random seed to use (default 42 = original run)
    # THESIS_RUN_TAG  : if set, results go to experiments/multiseed/ instead
    #                   of overwriting the canonical model results
    # THESIS_MAX_EPOCHS: cap training epochs (default 500 = original)
    SEED = int(os.environ.get("THESIS_SEED", "42"))
    RUN_TAG = os.environ.get("THESIS_RUN_TAG", "")
    MAX_EPOCHS = int(os.environ.get("THESIS_MAX_EPOCHS", "500"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================================
    # LOAD DATA
    # ======================================

    with open(os.path.join(PROJECT_ROOT, "data/processed/multiplex_graphs.pkl"), "rb") as f:
        graphs = pickle.load(f)

    with open(os.path.join(PROJECT_ROOT, "data/processed/train_edges.pkl"), "rb") as f:
        train_edges = pickle.load(f)

    with open(os.path.join(PROJECT_ROOT, "data/processed/train_negative_edges.pkl"), "rb") as f:
        train_negative_edges = pickle.load(f)

    with open(os.path.join(PROJECT_ROOT, "data/processed/val_edges.pkl"), "rb") as f:
        val_edges = pickle.load(f)

    with open(os.path.join(PROJECT_ROOT, "data/processed/val_negative_edges.pkl"), "rb") as f:
        val_negative_edges = pickle.load(f)

    with open(os.path.join(PROJECT_ROOT, "data/processed/test_edges.pkl"), "rb") as f:
        test_edges = pickle.load(f)

    with open(os.path.join(PROJECT_ROOT, "data/processed/test_negative_edges.pkl"), "rb") as f:
        test_negative_edges = pickle.load(f)

    layer_names = list(graphs.keys())

    # ======================================
    # GLOBAL NODE MAPPING (shared across ALL layers)
    # ======================================

    global_nodes = set()
    for g in graphs.values():
        global_nodes.update(g.nodes())
    global_nodes = sorted(global_nodes)

    global_node_mapping = {node: idx for idx, node in enumerate(global_nodes)}

    print("Total nodes across all layers (global):", len(global_nodes))

    # ======================================
    # BUILD PER-LAYER TRAINING DATA
    # ======================================

    layer_data = {}
    layer_samples = {}

    def map_edges(edges):
        mapped = []
        for edge in edges:
            u = edge[0]
            v = edge[1]
            mapped.append((global_node_mapping[u], global_node_mapping[v]))
        return mapped

    for layer_name in layer_names:

        print("Preparing layer:", layer_name)

        train_graph = create_training_graph(
            global_nodes,
            train_edges[layer_name]
        )

        data = convert_to_pyg(train_graph, global_node_mapping)
        layer_data[layer_name] = data.to(device)

        train_positive = map_edges(train_edges[layer_name])
        train_negative = map_edges(train_negative_edges[layer_name])

        val_positive = map_edges(val_edges[layer_name])
        val_negative = map_edges(val_negative_edges[layer_name])

        test_positive = map_edges(test_edges[layer_name])
        test_negative = map_edges(test_negative_edges[layer_name])

        train_labels = torch.tensor(
            [1] * len(train_positive) + [0] * len(train_negative),
            dtype=torch.float,
            device=device
        )

        layer_samples[layer_name] = {
            "train_samples": train_positive + train_negative,
            "train_labels": train_labels,
            "val_positive": val_positive,
            "val_negative": val_negative,
            "test_positive": test_positive,
            "test_negative": test_negative,
        }

    # ======================================
    # MODEL
    # ======================================

    input_dim = layer_data[layer_names[0]].x.shape[1]

    model = ProposedMultiplexModel(
        layer_names=layer_names,
        input_dim=input_dim,
        hidden_dim=128,
        embedding_dim=64,
        heads=8,
        dropout=0.4
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )

    criterion = nn.BCEWithLogitsLoss()

    # ======================================
    # JOINT TRAINING (all layers together)
    # ======================================

    best_avg_auc = 0
    patience = 30
    counter = 0
    best_state = None
    best_beta = None

    for epoch in range(MAX_EPOCHS):

        model.train()
        optimizer.zero_grad()

        embeddings, beta = model.encode(layer_data)

        total_loss = 0

        for layer_name in layer_names:

            samples = layer_samples[layer_name]

            scores = model.decode(
                embeddings,
                layer_name,
                samples["train_samples"]
            )

            loss = criterion(scores, samples["train_labels"])
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()

        # ----------------------------------
        # Validation (macro-average ROC-AUC across layers)
        # ----------------------------------

        model.eval()

        with torch.no_grad():

            embeddings, beta = model.encode(layer_data)

            layer_aucs = []

            for layer_name in layer_names:

                samples = layer_samples[layer_name]

                val_edges_l = samples["val_positive"] + samples["val_negative"]

                val_scores = model.decode(embeddings, layer_name, val_edges_l)
                val_scores = torch.sigmoid(val_scores).cpu().numpy()

                val_labels = (
                    [1] * len(samples["val_positive"])
                    + [0] * len(samples["val_negative"])
                )

                val_results = evaluate_all(val_labels, val_scores)
                layer_aucs.append(val_results["ROC-AUC"])

            avg_auc = float(np.mean(layer_aucs))

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.4f} | Avg Val ROC-AUC: {avg_auc:.4f}")

        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_beta = beta.detach().cpu().numpy()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping at epoch:", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ======================================
    # PER-LAYER BEST THRESHOLD (from VALIDATION set only)
    # ======================================
    #
    # Sigmoid outputs from a GNN are not necessarily calibrated the
    # same way as e.g. a LogisticRegression probability, so a fixed
    # 0.5 cutoff can under-report F1 even when ranking (ROC-AUC/PR-AUC)
    # is good. We tune the threshold PER LAYER using only validation
    # data (never test data) to keep this leakage-free.

    model.eval()

    best_thresholds = {}

    with torch.no_grad():

        embeddings, beta = model.encode(layer_data)

        for layer_name in layer_names:

            samples = layer_samples[layer_name]

            val_edges_l = samples["val_positive"] + samples["val_negative"]

            val_scores = model.decode(embeddings, layer_name, val_edges_l)
            val_scores = torch.sigmoid(val_scores).cpu().numpy()

            val_labels = (
                [1] * len(samples["val_positive"])
                + [0] * len(samples["val_negative"])
            )

            threshold, val_f1 = find_best_threshold(val_labels, val_scores)
            best_thresholds[layer_name] = threshold

            print(f"{layer_name}: best threshold = {threshold:.4f} (val F1 = {val_f1:.4f})")

    # ======================================
    # LEARNED LAYER IMPORTANCE (interpretability)
    # ======================================

    print("\nLearned layer importance weights (beta):")
    for layer_name, weight in zip(layer_names, best_beta):
        print(f"{layer_name}: {weight:.4f}")

    # ======================================
    # TESTING
    # ======================================

    model.eval()

    all_metrics = []
    all_predictions = []

    with torch.no_grad():

        embeddings, beta = model.encode(layer_data)

        for layer_name in layer_names:

            samples = layer_samples[layer_name]

            test_edges_l = samples["test_positive"] + samples["test_negative"]

            test_scores = model.decode(embeddings, layer_name, test_edges_l)
            test_scores = torch.sigmoid(test_scores).cpu().numpy()

            test_labels = (
                [1] * len(samples["test_positive"])
                + [0] * len(samples["test_negative"])
            )

            results = evaluate_all(test_labels, test_scores, threshold=best_thresholds[layer_name])
            results["Layer"] = layer_name
            results["Layer_Importance"] = float(best_beta[layer_names.index(layer_name)])
            results["Threshold_Used"] = best_thresholds[layer_name]

            all_metrics.append(results)

            print("\nTesting Results:", layer_name)
            for metric, value in results.items():
                if metric not in ("Layer",):
                    if isinstance(value, float):
                        print(f"{metric}: {value:.4f}")

            original_edges = test_edges[layer_name] + test_negative_edges[layer_name]

            for edge, score, label in zip(original_edges, test_scores, test_labels):
                all_predictions.append((
                    edge[0],
                    edge[1],
                    float(score),
                    label,
                    layer_name
                ))

    # ======================================
    # SAVE RESULTS
    # ======================================

    if RUN_TAG:
        # multi-seed run: keep canonical results untouched
        metrics_base = os.path.join(PROJECT_ROOT, "experiments/multiseed/metrics")
        predictions_base = os.path.join(PROJECT_ROOT, "experiments/multiseed/predictions")
        save_metrics(all_metrics, f"proposed_A_{RUN_TAG}.csv", base_folder=metrics_base)
        save_predictions(all_predictions, f"proposed_A_{RUN_TAG}.csv", base_folder=predictions_base)
    else:
        metrics_base = os.path.join(PROJECT_ROOT, "experiments/model_results/metrics")
        predictions_base = os.path.join(PROJECT_ROOT, "experiments/model_results/predictions")
        save_metrics(all_metrics, "proposed_model_metrics.csv", base_folder=metrics_base)
        save_predictions(all_predictions, "proposed_model_predictions.csv", base_folder=predictions_base)

    print("\nProposed attention-enhanced multiplex GNN completed successfully!")