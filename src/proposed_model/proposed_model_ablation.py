# ==========================================================
# ABLATION MODEL (component ablation study)
# ==========================================================
# This file does NOT modify proposed_model.py. It imports the
# original building blocks and data helpers from it, and defines
# a separate ablation model in which ONE component is removed at
# a time:
#   THESIS_ABLATION = "no_weaktie"   -> plain GAT encoder (no 1-Jaccard signal)
#   THESIS_ABLATION = "no_layerattn" -> cross-layer attention replaced by mean
#   THESIS_ABLATION = "no_fusion"    -> fusion gate replaced by residual sum
# Results are saved under abl_* tags, so proposed_A stays untouched.
# ==========================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.append(PROJECT_ROOT)

import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from src.evaluation.metrics import evaluate_all, find_best_threshold
from src.utils.save_results import save_metrics, save_predictions

# reuse the ORIGINAL building blocks + data helpers (no edits to that file)
from src.proposed_model.proposed_model import (
    create_training_graph,
    convert_to_pyg,
    WeakTieGATEncoder,
    LayerAttention,
    FusionGate,
    EdgeDecoder,
)


# ----------------------------------------------------------
# Plain GAT encoder (used only for the "no_weaktie" ablation):
# identical to WeakTieGATEncoder but WITHOUT the edge signal.
# ----------------------------------------------------------
class PlainGATEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=32, heads=8, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1,
                             concat=False, dropout=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)          # edge_attr intentionally ignored
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ----------------------------------------------------------
# Ablation model
# ----------------------------------------------------------
class AblationMultiplexModel(nn.Module):

    def __init__(self, layer_names, input_dim, hidden_dim=64,
                 embedding_dim=32, heads=8, dropout=0.5, ablation="none"):
        super().__init__()
        self.layer_names = layer_names
        self.ablation = ablation
        self.safe_keys = {ln: ln.replace(".", "__DOT__") for ln in layer_names}

        Encoder = PlainGATEncoder if ablation == "no_weaktie" else WeakTieGATEncoder
        self.encoders = nn.ModuleDict({
            self.safe_keys[ln]: Encoder(
                input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=embedding_dim, heads=heads, dropout=dropout)
            for ln in layer_names
        })

        self.layer_attention = LayerAttention(embedding_dim)
        self.fusion_gate = FusionGate(embedding_dim)
        self.decoder = EdgeDecoder(embedding_dim)

    def encode(self, layer_data):
        layer_embeddings = []
        for ln in self.layer_names:
            data = layer_data[ln]
            z = self.encoders[self.safe_keys[ln]](
                data.x, data.edge_index, data.edge_attr)
            layer_embeddings.append(z)

        if self.ablation == "no_layerattn":
            Z = torch.stack(layer_embeddings, dim=0)      # [L, N, D]
            z_fused = Z.mean(dim=0)                        # uniform mean
            L = len(self.layer_names)
            beta = torch.full((L,), 1.0 / L, device=z_fused.device)
        else:
            z_fused, beta = self.layer_attention(layer_embeddings)

        final_embeddings = {}
        for ln, z_local in zip(self.layer_names, layer_embeddings):
            if self.ablation == "no_fusion":
                final_embeddings[ln] = z_local + z_fused  # simple residual sum
            else:
                final_embeddings[ln] = self.fusion_gate(z_local, z_fused)
        return final_embeddings, beta

    def decode(self, embeddings, layer_name, edges):
        return self.decoder(embeddings[layer_name], edges)


# ----------------------------------------------------------
# MAIN (same training/eval as the full model)
# ----------------------------------------------------------
if __name__ == "__main__":

    SEED = int(os.environ.get("THESIS_SEED", "42"))
    RUN_TAG = os.environ.get("THESIS_RUN_TAG", "")
    MAX_EPOCHS = int(os.environ.get("THESIS_MAX_EPOCHS", "500"))
    ABLATION = os.environ.get("THESIS_ABLATION", "no_weaktie")

    MODEL_TAG = {
        "no_weaktie": "abl_no_weaktie",
        "no_layerattn": "abl_no_layerattn",
        "no_fusion": "abl_no_fusion",
    }.get(ABLATION, "abl_unknown")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Ablation mode: {ABLATION}  (save tag: {MODEL_TAG})")

    D = os.path.join(PROJECT_ROOT, "data/processed")
    with open(os.path.join(D, "multiplex_graphs.pkl"), "rb") as f:
        graphs = pickle.load(f)
    with open(os.path.join(D, "train_edges.pkl"), "rb") as f:
        train_edges = pickle.load(f)
    with open(os.path.join(D, "train_negative_edges.pkl"), "rb") as f:
        train_negative_edges = pickle.load(f)
    with open(os.path.join(D, "val_edges.pkl"), "rb") as f:
        val_edges = pickle.load(f)
    with open(os.path.join(D, "val_negative_edges.pkl"), "rb") as f:
        val_negative_edges = pickle.load(f)
    with open(os.path.join(D, "test_edges.pkl"), "rb") as f:
        test_edges = pickle.load(f)
    with open(os.path.join(D, "test_negative_edges.pkl"), "rb") as f:
        test_negative_edges = pickle.load(f)

    layer_names = list(graphs.keys())

    global_nodes = set()
    for g in graphs.values():
        global_nodes.update(g.nodes())
    global_nodes = sorted(global_nodes)
    global_node_mapping = {node: idx for idx, node in enumerate(global_nodes)}
    print("Total global nodes:", len(global_nodes))

    layer_data = {}
    layer_samples = {}


    def map_edges(edges):
        return [(global_node_mapping[e[0]], global_node_mapping[e[1]]) for e in edges]

    for layer_name in layer_names:
        print("Preparing layer:", layer_name)
        train_graph = create_training_graph(global_nodes, train_edges[layer_name])
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
            dtype=torch.float, device=device)

        layer_samples[layer_name] = {
            "train_samples": train_positive + train_negative,
            "train_labels": train_labels,
            "val_positive": val_positive, "val_negative": val_negative,
            "test_positive": test_positive, "test_negative": test_negative,
        }

    input_dim = layer_data[layer_names[0]].x.shape[1]

    model = AblationMultiplexModel(
        layer_names=layer_names, input_dim=input_dim,
        hidden_dim=128, embedding_dim=64, heads=8, dropout=0.4,
        ablation=ABLATION).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

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
            scores = model.decode(embeddings, layer_name, samples["train_samples"])
            total_loss = total_loss + criterion(scores, samples["train_labels"])
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings, beta = model.encode(layer_data)
            layer_aucs = []
            for layer_name in layer_names:
                samples = layer_samples[layer_name]
                val_edges_l = samples["val_positive"] + samples["val_negative"]
                val_scores = torch.sigmoid(
                    model.decode(embeddings, layer_name, val_edges_l)).cpu().numpy()
                val_labels = [1] * len(samples["val_positive"]) + [0] * len(samples["val_negative"])
                layer_aucs.append(evaluate_all(val_labels, val_scores)["ROC-AUC"])
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

    model.eval()
    best_thresholds = {}
    with torch.no_grad():
        embeddings, beta = model.encode(layer_data)
        for layer_name in layer_names:
            samples = layer_samples[layer_name]
            val_edges_l = samples["val_positive"] + samples["val_negative"]
            val_scores = torch.sigmoid(
                model.decode(embeddings, layer_name, val_edges_l)).cpu().numpy()
            val_labels = [1] * len(samples["val_positive"]) + [0] * len(samples["val_negative"])
            threshold, _ = find_best_threshold(val_labels, val_scores)
            best_thresholds[layer_name] = threshold

    model.eval()
    all_metrics = []
    all_predictions = []
    with torch.no_grad():
        embeddings, beta = model.encode(layer_data)
        for layer_name in layer_names:
            samples = layer_samples[layer_name]
            test_edges_l = samples["test_positive"] + samples["test_negative"]
            test_scores = torch.sigmoid(
                model.decode(embeddings, layer_name, test_edges_l)).cpu().numpy()
            test_labels = [1] * len(samples["test_positive"]) + [0] * len(samples["test_negative"])

            results = evaluate_all(test_labels, test_scores, threshold=best_thresholds[layer_name])
            results["Layer"] = layer_name
            results["Layer_Importance"] = float(best_beta[layer_names.index(layer_name)])
            results["Threshold_Used"] = best_thresholds[layer_name]
            all_metrics.append(results)

            original_edges = test_edges[layer_name] + test_negative_edges[layer_name]
            for edge, score, label in zip(original_edges, test_scores, test_labels):
                all_predictions.append((edge[0], edge[1], float(score), label, layer_name))

    if RUN_TAG:
        metrics_base = os.path.join(PROJECT_ROOT, "experiments/multiseed/metrics")
        predictions_base = os.path.join(PROJECT_ROOT, "experiments/multiseed/predictions")
    else:
        metrics_base = os.path.join(PROJECT_ROOT, "experiments/ablation_results/metrics")
        predictions_base = os.path.join(PROJECT_ROOT, "experiments/ablation_results/predictions")
        RUN_TAG = "single"

    save_metrics(all_metrics, f"{MODEL_TAG}_{RUN_TAG}.csv", base_folder=metrics_base)
    save_predictions(all_predictions, f"{MODEL_TAG}_{RUN_TAG}.csv", base_folder=predictions_base)

    print(f"\nAblation '{ABLATION}' completed successfully!")