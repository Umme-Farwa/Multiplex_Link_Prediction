# ==========================================
# IMPORT PATH FIX
# ==========================================

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../.."
        )
    )
)

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
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from src.evaluation.metrics import evaluate_all, find_best_threshold
from src.utils.save_results import (
    save_metrics,
    save_predictions
)


# ==========================================
# Create Training Graph
# ==========================================

def create_training_graph(original_graph, train_edges):
    train_graph = nx.Graph()
    train_graph.add_nodes_from(original_graph.nodes())
    train_graph.add_edges_from(train_edges)
    return train_graph


# ==========================================
# Structural Node Features
# ==========================================

def create_node_features(graph):

    print("Generating node features...")

    nodes = sorted(graph.nodes())
    num_nodes = len(nodes)

    degree_centrality = nx.degree_centrality(graph)
    pagerank = nx.pagerank(graph)
    clustering = nx.clustering(graph)

    try:
        kcore = nx.core_number(graph)
    except Exception:
        kcore = {node: 0 for node in nodes}

    # Betweenness centrality: how often a node lies on shortest paths
    # between other nodes -- captures a "bridging" structural role.
    # Sampled (k pivots) for speed, matching the proposed model.
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
    # other important nodes -- a "global influence" signal.
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

    # --- Normalization (z-score) ---
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    x = (x - mean) / (std + 1e-8)

    return x

# ==========================================
# Convert Graph to PyG Data
# ==========================================

def convert_to_pyg(graph):

    nodes = sorted(graph.nodes())
    node_mapping = {node: index for index, node in enumerate(nodes)}

    edge_list = []
    for u, v in graph.edges():
        edge_list.append([node_mapping[u], node_mapping[v]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Symmetrize edges so GCN message passing works both directions
    if edge_index.numel() > 0:
        edge_index = to_undirected(edge_index)

    x = create_node_features(graph)

    data = Data(x=x, edge_index=edge_index)

    return data, node_mapping


# ==========================================
# GCN Encoder
# ==========================================

class GCNEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ==========================================
# MLP Edge Decoder (vectorized)
# ==========================================

class EdgeDecoder(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embeddings, edges):
        if not torch.is_tensor(edges):
            edges = torch.tensor(edges, dtype=torch.long, device=embeddings.device)

        u_emb = embeddings[edges[:, 0]]
        v_emb = embeddings[edges[:, 1]]

        pairs = torch.cat([u_emb, v_emb], dim=1)

        scores = self.decoder(pairs).squeeze(-1)

        return scores


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # --- multi-seed support (backward compatible) ---
    # THESIS_SEED     : which random seed to use (default 42 = original run)
    # THESIS_RUN_TAG  : if set, results go to experiments/multiseed/ instead
    #                   of overwriting the canonical baseline results
    # THESIS_MAX_EPOCHS: cap training epochs (default 200 = original)
    SEED = int(os.environ.get("THESIS_SEED", "42"))
    RUN_TAG = os.environ.get("THESIS_RUN_TAG", "")
    MAX_EPOCHS = int(os.environ.get("THESIS_MAX_EPOCHS", "200"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================================
    # LOAD DATA
    # ======================================

    with open("data/processed/multiplex_graphs.pkl", "rb") as f:
        graphs = pickle.load(f)

    with open("data/processed/train_edges.pkl", "rb") as f:
        train_edges = pickle.load(f)

    with open("data/processed/train_negative_edges.pkl", "rb") as f:
        train_negative_edges = pickle.load(f)

    with open("data/processed/val_edges.pkl", "rb") as f:
        val_edges = pickle.load(f)

    with open("data/processed/val_negative_edges.pkl", "rb") as f:
        val_negative_edges = pickle.load(f)

    with open("data/processed/test_edges.pkl", "rb") as f:
        test_edges = pickle.load(f)

    with open("data/processed/test_negative_edges.pkl", "rb") as f:
        test_negative_edges = pickle.load(f)

    all_metrics = []
    all_predictions = []

    # ======================================
    # PROCESS EACH LAYER
    # ======================================

    for layer_name, graph in graphs.items():

        print("\n====================")
        print("Layer:", layer_name)
        print("====================")

        train_graph = create_training_graph(graph, train_edges[layer_name])

        data, node_mapping = convert_to_pyg(train_graph)
        data = data.to(device)

        # -------------------------------
        # Map node ids
        # -------------------------------

        def map_edges(edges):
            mapped = []
            for edge in edges:
                u = edge[0]
                v = edge[1]
                mapped.append((node_mapping[u], node_mapping[v]))
            return mapped

        # -------------------------------
        # Prepare samples
        # -------------------------------

        train_positive = map_edges(train_edges[layer_name])
        train_negative = map_edges(train_negative_edges[layer_name])

        val_positive = map_edges(val_edges[layer_name])
        val_negative = map_edges(val_negative_edges[layer_name])

        test_positive = map_edges(test_edges[layer_name])
        test_negative = map_edges(test_negative_edges[layer_name])

        train_samples = train_positive + train_negative

        train_labels = torch.tensor(
            [1] * len(train_positive) + [0] * len(train_negative),
            dtype=torch.float,
            device=device
        )

        # ==================================
        # MODEL
        # ==================================

        encoder = GCNEncoder(
            input_dim=data.x.shape[1],
            hidden_dim=64,
            output_dim=32
        ).to(device)

        decoder = EdgeDecoder(embedding_dim=32).to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=0.001,
            weight_decay=5e-4
        )

        criterion = nn.BCEWithLogitsLoss()

        # ==================================
        # TRAINING + VALIDATION
        # ==================================

        best_auc = 0
        patience = 20
        counter = 0
        best_encoder = None
        best_decoder = None

        for epoch in range(MAX_EPOCHS):

            encoder.train()
            decoder.train()

            optimizer.zero_grad()

            embeddings = encoder(data.x, data.edge_index)

            train_scores = decoder(embeddings, train_samples)

            loss = criterion(train_scores, train_labels)

            loss.backward()
            optimizer.step()

            # ------------------------------
            # Validation
            # ------------------------------

            encoder.eval()
            decoder.eval()

            with torch.no_grad():

                embeddings = encoder(data.x, data.edge_index)

                val_samples = val_positive + val_negative

                val_scores = decoder(embeddings, val_samples)
                val_scores = torch.sigmoid(val_scores).cpu().numpy()

            val_labels = [1] * len(val_positive) + [0] * len(val_negative)

            val_results = evaluate_all(val_labels, val_scores)
            val_auc = val_results["ROC-AUC"]

            if val_auc > best_auc:
                best_auc = val_auc
                best_encoder = encoder.state_dict()
                best_decoder = decoder.state_dict()
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping at epoch:", epoch)
                break

        # Restore best model
        if best_encoder is not None:
            encoder.load_state_dict(best_encoder)
            decoder.load_state_dict(best_decoder)

        # ==================================
        # BEST THRESHOLD (from VALIDATION set only)
        # ==================================

        encoder.eval()
        decoder.eval()

        with torch.no_grad():

            embeddings = encoder(data.x, data.edge_index)

            val_samples = val_positive + val_negative

            val_scores_final = decoder(embeddings, val_samples)
            val_scores_final = torch.sigmoid(val_scores_final).cpu().numpy()

        val_labels_final = [1] * len(val_positive) + [0] * len(val_negative)

        best_threshold, best_val_f1 = find_best_threshold(val_labels_final, val_scores_final)

        print(f"Best threshold (from val): {best_threshold:.4f} (val F1 = {best_val_f1:.4f})")

        # ==================================
        # TESTING
        # ==================================

        encoder.eval()
        decoder.eval()

        with torch.no_grad():

            embeddings = encoder(data.x, data.edge_index)

            test_samples = test_positive + test_negative

            test_scores = decoder(embeddings, test_samples)
            test_scores = torch.sigmoid(test_scores).cpu().numpy()

        test_labels = [1] * len(test_positive) + [0] * len(test_negative)

        results = evaluate_all(test_labels, test_scores, threshold=best_threshold)
        results["Layer"] = layer_name
        results["Threshold_Used"] = best_threshold

        all_metrics.append(results)

        print("\nTesting Results:")
        for metric, value in results.items():
            if metric != "Layer":
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")

        # ==================================
        # SAVE PREDICTIONS
        # ==================================

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
        save_metrics(all_metrics, f"gcn_{RUN_TAG}.csv", base_folder="experiments/multiseed/metrics")
        save_predictions(all_predictions, f"gcn_{RUN_TAG}.csv", base_folder="experiments/multiseed/predictions")
    else:
        save_metrics(all_metrics, "gnn/gcn_metrics.csv")
        save_predictions(all_predictions, "gnn/gcn_predictions.csv")

    print("\nImproved GCN baseline completed successfully!")