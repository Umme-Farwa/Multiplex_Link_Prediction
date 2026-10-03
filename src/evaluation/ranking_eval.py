# ==========================================================
# RANKING EVALUATION  --  Hits@K  and  MRR
# ==========================================================
#
# WHY THIS EXISTS
# ---------------
# ROC-AUC / PR-AUC / F1 answer "yes-or-no" quality. Hits@K and MRR
# answer a RANKING question: "when the model ranks candidate links,
# does the TRUE link land near the top?" Link-prediction papers
# almost always report these, so we add them for completeness.
#
# NO RE-TRAINING NEEDED
# ---------------------
# These metrics are computed purely from the SCORES already saved in
# each model's prediction CSV (experiments/.../*_predictions.csv).
# We never re-train -- we just re-read the saved scores. This is why
# adding these metrics did not require re-running any seeds.
#
# THE PROTOCOL (what we actually measure)
# ---------------------------------------
# Each prediction file holds, per layer, the TEST positive edges
# (label=1) and TEST negative edges (label=0), each with a score.
# For every POSITIVE edge we rank it against ALL the negative edges
# in the SAME layer:
#
#     rank = (# negatives scored higher than the positive) + 1
#            (+ half the ties, so equal scores are handled fairly)
#
#   * MRR   = mean of 1/rank over all positives   (higher = better)
#   * Hits@K = fraction of positives whose rank <= K (higher = better)
#
# We compute this per layer, then macro-average across the 13 layers
# (same averaging scheme used for the other metrics), so every model
# gets one comparable number.
#
# Tie handling matters here: heuristics like Adamic-Adar give score 0
# to BOTH the positive and the negatives on bridge links -> big ties
# -> a middling rank. That is CORRECT: it honestly reflects that such
# heuristics cannot rank bridges. So this metric tells the true story.
#
# USAGE (run from project root, using the .venv python for consistency)
# ---------------------------------------------------------------------
#   .\.venv\Scripts\python.exe src/evaluation/ranking_eval.py
# ==========================================================

import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

# The K values for Hits@K. 1, 3, 5, 10 are the usual choices.
K_VALUES = [1, 3, 5, 10]

# model name -> canonical prediction CSV (relative to project root)
MODEL_PREDICTIONS = {
    "Proposed":               "experiments/model_results/predictions/proposed_model_predictions.csv",
    "GAT":                    "experiments/baseline_results/predictions/gnn/gat_predictions.csv",
    "GCN":                    "experiments/baseline_results/predictions/gnn/gcn_predictions.csv",
    "Node2Vec":               "experiments/baseline_results/predictions/embedding/node2vec_predictions.csv",
    "Adamic-Adar":            "experiments/baseline_results/predictions/classical/adamic_adar_predictions.csv",
    "Common Neighbors":       "experiments/baseline_results/predictions/classical/common_neighbors_predictions.csv",
    "Jaccard":                "experiments/baseline_results/predictions/classical/jaccard_predictions.csv",
    "Preferential Attachment":"experiments/baseline_results/predictions/classical/preferential_attachment_predictions.csv",
}


def rank_positive_edges(pos_scores, neg_scores):
    """Given the scores of positive edges and negative edges (in one
    layer), return the rank of every positive edge against the pool of
    negatives, with fair average-tie handling.

    rank = (# negatives strictly greater) + 1 + (# tied negatives)/2
    """

    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_sorted = np.sort(np.asarray(neg_scores, dtype=float))

    n_neg = len(neg_sorted)
    if n_neg == 0:
        # no negatives to rank against -> every positive is rank 1
        return np.ones_like(pos_scores)

    # for each positive score s:
    #   left  = # negatives with score <  s
    #   right = # negatives with score <= s
    left = np.searchsorted(neg_sorted, pos_scores, side="left")
    right = np.searchsorted(neg_sorted, pos_scores, side="right")

    greater = n_neg - right     # negatives strictly greater than s
    equal = right - left        # negatives exactly equal to s (ties)

    rank = greater + 1 + equal / 2.0
    return rank


def evaluate_file(path):
    """Compute macro-averaged MRR and Hits@K for one prediction CSV.
    Returns a dict, or None if the file is missing."""

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    # collect per-layer metrics, then average across layers
    layer_mrr = []
    layer_hits = {k: [] for k in K_VALUES}

    for layer_name, group in df.groupby("layer"):

        pos = group[group["label"] == 1]["score"].values
        neg = group[group["label"] == 0]["score"].values

        if len(pos) == 0:
            continue

        ranks = rank_positive_edges(pos, neg)

        layer_mrr.append(float(np.mean(1.0 / ranks)))

        for k in K_VALUES:
            layer_hits[k].append(float(np.mean(ranks <= k)))

    if len(layer_mrr) == 0:
        return None

    result = {"MRR": float(np.mean(layer_mrr))}
    for k in K_VALUES:
        result[f"Hits@{k}"] = float(np.mean(layer_hits[k]))
    return result


def main():

    rows = []

    for model_name, rel_path in MODEL_PREDICTIONS.items():

        path = os.path.join(PROJECT_ROOT, rel_path)
        res = evaluate_file(path)

        if res is None:
            print(f"  (skipped {model_name}: prediction file not found)")
            continue

        row = {"Model": model_name}
        row.update(res)
        rows.append(row)

    if not rows:
        print("No prediction files found. Did the models run?")
        return

    table = pd.DataFrame(rows)

    # order columns nicely
    cols = ["Model", "MRR"] + [f"Hits@{k}" for k in K_VALUES]
    table = table[cols]

    # sort by MRR (best first)
    table = table.sort_values("MRR", ascending=False).reset_index(drop=True)

    # ---- pretty print ----
    print("\n" + "=" * 70)
    print("RANKING METRICS  (macro-averaged over 13 layers; higher = better)")
    print("=" * 70)
    header = f"{'Model':<26}{'MRR':>8}" + "".join(f"{f'Hits@{k}':>9}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        line = f"{r['Model']:<26}{r['MRR']:>8.4f}"
        line += "".join(f"{r[f'Hits@{k}']:>9.4f}" for k in K_VALUES)
        print(line)

    # ---- save ----
    out_dir = os.path.join(PROJECT_ROOT, "experiments/summary")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ranking_metrics.csv")
    table.to_csv(out_path, index=False)

    print("\nSaved:", os.path.relpath(out_path, PROJECT_ROOT))
    print("Done.")


if __name__ == "__main__":
    main()
