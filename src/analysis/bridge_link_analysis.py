# ==========================================
# BRIDGE-LINK STRUCTURAL ANALYSIS  (v2)
# Two angles:
#   (A) Coarse community-based  (cross vs intra)
#   (B) Weak-tie based          (bridge = zero common neighbors)
# Post-hoc: reads saved predictions, no model re-run.
# ==========================================

import os
import sys
import pickle

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import roc_auc_score, average_precision_score


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

# Coarser Louvain -> fewer, larger communities (default is 1.0; lower = coarser)
COARSE_RESOLUTION = 0.3


MODELS = {
    "adamic_adar":                "experiments/baseline_results/predictions/classical/adamic_adar_predictions.csv",
    "common_neighbors":           "experiments/baseline_results/predictions/classical/common_neighbors_predictions.csv",
    "jaccard":                    "experiments/baseline_results/predictions/classical/jaccard_predictions.csv",
    "preferential_attachment":    "experiments/baseline_results/predictions/classical/preferential_attachment_predictions.csv",
    "node2vec":                   "experiments/baseline_results/predictions/embedding/node2vec_predictions.csv",
    "gcn":                        "experiments/baseline_results/predictions/gnn/gcn_predictions.csv",
    "gat":                        "experiments/baseline_results/predictions/gnn/gat_predictions.csv",
    "proposed_A":                 "experiments/model_results/predictions/proposed_model_predictions.csv",
    "proposed_B_weaktie_decoder": "experiments/ablation_results/predictions/proposed_weaktie_decoder_predictions.csv",
}


def build_train_graph(graph, train_edges_layer):
    tg = nx.Graph()
    tg.add_nodes_from(graph.nodes())
    for u, v, weight in train_edges_layer:
        tg.add_edge(u, v)
    return tg


def detect_communities(train_graph, resolution):
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(train_graph, resolution=resolution, seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(train_graph))

    node_to_comm = {}
    for i, community in enumerate(comms):
        for node in community:
            node_to_comm[node] = i
    return node_to_comm, len(comms)


def jaccard(train_graph, u, v):
    nu = set(train_graph.neighbors(u)) if train_graph.has_node(u) else set()
    nv = set(train_graph.neighbors(v)) if train_graph.has_node(v) else set()
    union = nu | nv
    if len(union) == 0:
        return 0.0
    return len(nu & nv) / len(union)


def key_of(layer, u, v):
    a, b = (u, v) if u <= v else (v, u)
    return (layer, a, b)


def macro_auc_table(group_map, group_names, title):
    """Per-model, per-group macro-averaged ROC-AUC / PR-AUC over layers."""

    rows = []

    for model_name, rel_path in MODELS.items():

        path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(path):
            print(f"[SKIP] {model_name}: {rel_path} not found")
            continue

        df = pd.read_csv(path)
        df["node1"] = df["node1"].astype(int)
        df["node2"] = df["node2"].astype(int)
        df["score"] = df["score"].astype(float)
        df["label"] = df["label"].astype(int)

        df["group"] = [
            group_map.get(key_of(l, u, v), group_names[0])
            for l, u, v in zip(df["layer"], df["node1"], df["node2"])
        ]

        for group in group_names:
            aucs, praucs, n_edges = [], [], 0
            for layer_name in df["layer"].unique():
                sub = df[(df["layer"] == layer_name) & (df["group"] == group)]
                if len(sub) == 0:
                    continue
                n_edges += len(sub)
                labels = sub["label"].values
                scores = sub["score"].values
                if len(set(labels)) < 2:
                    continue
                aucs.append(roc_auc_score(labels, scores))
                praucs.append(average_precision_score(labels, scores))

            rows.append({
                "Model": model_name,
                "Group": group,
                "Avg_ROC_AUC": float(np.mean(aucs)) if aucs else float("nan"),
                "Avg_PR_AUC": float(np.mean(praucs)) if praucs else float("nan"),
                "Layers_used": len(aucs),
                "Total_edges": n_edges,
            })

    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="Model", columns="Group", values="Avg_ROC_AUC")
    pivot = pivot.sort_values(group_names[0], ascending=False)

    print(f"\n================ {title} ================")
    print("(higher on the first column = better at the harder link type)\n")
    print(pivot.round(4).to_string())

    return summary, pivot


if __name__ == "__main__":

    with open(os.path.join(PROJECT_ROOT, "data/processed/multiplex_graphs.pkl"), "rb") as f:
        graphs = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, "data/processed/train_edges.pkl"), "rb") as f:
        train_edges = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, "data/processed/test_edges.pkl"), "rb") as f:
        test_edges = pickle.load(f)
    with open(os.path.join(PROJECT_ROOT, "data/processed/test_negative_edges.pkl"), "rb") as f:
        test_negative_edges = pickle.load(f)

    layer_names = list(graphs.keys())

    # ---- Precompute both groupings for every test edge (same test set for all models) ----

    comm_group = {}     # key -> "cross"/"intra"
    weaktie_group = {}  # key -> "bridge"/"strong"

    comm_counts = {"cross": 0, "intra": 0}
    wt_counts = {"bridge": 0, "strong": 0}

    print(f"Building groupings (coarse community resolution = {COARSE_RESOLUTION})...\n")

    for layer_name in layer_names:

        tg = build_train_graph(graphs[layer_name], train_edges[layer_name])
        node2comm, n_comms = detect_communities(tg, COARSE_RESOLUTION)

        print(f"  {layer_name:20s} | nodes={tg.number_of_nodes():6d} | communities={n_comms}")

        all_test = list(test_edges[layer_name]) + list(test_negative_edges[layer_name])

        for edge in all_test:
            u, v = edge[0], edge[1]
            k = key_of(layer_name, u, v)

            cu, cv = node2comm.get(u), node2comm.get(v)
            g = "intra" if (cu is not None and cu == cv) else "cross"
            comm_group[k] = g
            comm_counts[g] += 1

            jac = jaccard(tg, u, v)
            wg = "bridge" if jac == 0.0 else "strong"
            weaktie_group[k] = wg
            wt_counts[wg] += 1

    # ---- Report splits ----

    ct = comm_counts["cross"] + comm_counts["intra"]
    wt = wt_counts["bridge"] + wt_counts["strong"]

    print("\n----- Split sizes -----")
    print(f"Community  | cross={comm_counts['cross']} ({100*comm_counts['cross']/ct:.1f}%)  intra={comm_counts['intra']} ({100*comm_counts['intra']/ct:.1f}%)")
    print(f"Weak-tie   | bridge={wt_counts['bridge']} ({100*wt_counts['bridge']/wt:.1f}%)  strong={wt_counts['strong']} ({100*wt_counts['strong']/wt:.1f}%)")

    # ---- Table A: coarse community ----

    summ_a, pivot_a = macro_auc_table(comm_group, ["cross", "intra"],
                                      f"ANGLE A: COARSE COMMUNITY (res={COARSE_RESOLUTION})")

    # ---- Table B: weak-tie ----

    summ_b, pivot_b = macro_auc_table(weaktie_group, ["bridge", "strong"],
                                      "ANGLE B: WEAK-TIE (bridge = zero common neighbors)")

    # ---- Save ----

    out_dir = os.path.join(PROJECT_ROOT, "experiments/analysis")
    os.makedirs(out_dir, exist_ok=True)

    pivot_a.round(4).to_csv(os.path.join(out_dir, "bridge_by_community_coarse.csv"))
    pivot_b.round(4).to_csv(os.path.join(out_dir, "bridge_by_weaktie.csv"))
    summ_a.to_csv(os.path.join(out_dir, "bridge_by_community_coarse_long.csv"), index=False)
    summ_b.to_csv(os.path.join(out_dir, "bridge_by_weaktie_long.csv"), index=False)

    print("\nSaved 4 files in experiments/analysis/")
    print("Bridge-link analysis (v2) completed!")