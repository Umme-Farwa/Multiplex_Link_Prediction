# ==========================================================
# MAKE RESULTS  --  all thesis figures + tables in ONE run
# ==========================================================
#
# WHAT THIS DOES
# --------------
# Reads every result that has already been produced (no re-training,
# no re-running models) and generates, in one go:
#
#   TABLES  -> experiments/summary/tables/   (CSV, ready for the thesis)
#     * main_metrics_table.csv     ROC-AUC / PR-AUC / F1 (mean +/- std)
#     * ranking_table.csv          MRR / Hits@K
#     * bridge_table.csv           bridge vs strong ROC-AUC
#     * significance_table.csv     proposed vs GAT/GCN, p-values
#
#   FIGURES -> experiments/summary/figures/  (PNG, 200 dpi, for slides + thesis)
#     * fig1_roc_auc.png           ROC-AUC bar chart with error bars
#     * fig2_all_metrics.png       ROC / PR / F1 grouped bars
#     * fig3_bridge_vs_strong.png  the KEY finding (heuristics = 0.5 on bridges)
#     * fig4_layer_importance.png  interpretable beta weight per layer
#     * fig5_ranking.png           MRR + Hits@10
#
# DATA SOURCES (all pre-existing)
#   experiments/multiseed/multiseed_summary.csv   (neural: mean/std over seeds)
#   experiments/multiseed/multiseed_per_seed.csv  (for significance tests)
#   experiments/baseline_results/metrics/classical/*_metrics.csv (deterministic)
#   experiments/summary/ranking_metrics.csv
#   experiments/analysis/bridge_by_weaktie.csv
#   experiments/model_results/metrics/proposed_model_metrics.csv (beta)
#
# USAGE (from project root, using the .venv python)
#   .\.venv\Scripts\python.exe src/experiments/make_results.py
# ==========================================================

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # no display needed -> just save PNGs
import matplotlib.pyplot as plt

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

FIG_DIR = os.path.join(PROJECT_ROOT, "experiments/summary/figures")
TAB_DIR = os.path.join(PROJECT_ROOT, "experiments/summary/tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ---- canonical model identities ----
# maps the many spellings used across files -> one canonical key
def norm(name):
    n = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if n.startswith("proposed"):
        return "proposed"
    return n

DISPLAY = {
    "proposed": "Proposed",
    "gat": "GAT",
    "gcn": "GCN",
    "node2vec": "Node2Vec",
    "adamic_adar": "Adamic-Adar",
    "common_neighbors": "Common Neighbors",
    "jaccard": "Jaccard",
    "preferential_attachment": "Preferential Attachment",
}

PROPOSED_KEYS = {"proposed"}

CLASSICAL_FILES = {
    "adamic_adar": "experiments/baseline_results/metrics/classical/adamic_adar_metrics.csv",
    "common_neighbors": "experiments/baseline_results/metrics/classical/common_neighbors_metrics.csv",
    "jaccard": "experiments/baseline_results/metrics/classical/jaccard_metrics.csv",
    "preferential_attachment": "experiments/baseline_results/metrics/classical/preferential_attachment_metrics.csv",
}

METRICS = ["ROC-AUC", "PR-AUC", "F1-score"]

# highlight colour for the proposed model vs everything else
C_PROP = "#0E7C86"     # teal   -> proposed
C_BASE = "#9AA3B2"     # grey   -> baselines
C_PR = "#B4690E"       # amber  -> secondary metric
C_F1 = "#3B6FB0"       # blue   -> third metric


def _p(rel):
    return os.path.join(PROJECT_ROOT, rel)


def _exists(rel):
    return os.path.exists(_p(rel))


def bar_colors(keys):
    return [C_PROP if k in PROPOSED_KEYS else C_BASE for k in keys]


def annotate(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom", fontsize=8)


# ==========================================================
# 1. LOAD everything into one place
# ==========================================================

# ---- neural models: mean +/- std from multi-seed ----
neural = {}   # key -> {metric: (mean, std)}
if _exists("experiments/multiseed/multiseed_summary.csv"):
    ms = pd.read_csv(_p("experiments/multiseed/multiseed_summary.csv"))
    for _, r in ms.iterrows():
        k = norm(r["model"])
        neural.setdefault(k, {})[r["metric"]] = (float(r["mean"]), float(r["std"]))

# ---- classical models: single deterministic value (avg over layers) ----
classical = {}   # key -> {metric: mean}
for k, rel in CLASSICAL_FILES.items():
    if _exists(rel):
        df = pd.read_csv(_p(rel))
        classical[k] = {m: float(df[m].mean()) for m in METRICS if m in df.columns}


# ==========================================================
# 2. MAIN METRICS TABLE  (ROC-AUC / PR-AUC / F1, mean +/- std)
# ==========================================================

def fmt_cell(mean, std):
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"

rows = []
# order: proposed first, then neural baselines, then classical
order = ["proposed", "node2vec", "gat", "gcn",
         "adamic_adar", "common_neighbors", "jaccard", "preferential_attachment"]

for k in order:
    row = {"Model": DISPLAY.get(k, k)}
    for m in METRICS:
        if k in neural and m in neural[k]:
            mean, std = neural[k][m]
            row[m] = fmt_cell(mean, std)
        elif k in classical and m in classical[k]:
            row[m] = fmt_cell(classical[k][m], None)
        else:
            row[m] = "-"
    rows.append(row)

main_table = pd.DataFrame(rows)
main_table.to_csv(os.path.join(TAB_DIR, "main_metrics_table.csv"), index=False)
print("Saved table: main_metrics_table.csv")
print(main_table.to_string(index=False))
print()


# ==========================================================
# 3. SIGNIFICANCE TABLE  (proposed vs GAT/GCN, paired t-test)
# ==========================================================

if _exists("experiments/multiseed/multiseed_per_seed.csv") and HAVE_SCIPY:
    ps = pd.read_csv(_p("experiments/multiseed/multiseed_per_seed.csv"))

    def seed_vals(model_key, metric):
        sub = ps[(ps["model"].map(norm) == model_key) & (ps["metric"] == metric)]
        return sub.sort_values("seed")["score"].values

    sig_rows = []
    for metric in METRICS:
        for prop in ("proposed",):
            for base in ("gat", "gcn"):
                a = seed_vals(prop, metric)
                b = seed_vals(base, metric)
                n = min(len(a), len(b))
                if n < 2:
                    continue
                a, b = a[:n], b[:n]
                delta = a.mean() - b.mean()
                p = float(stats.ttest_rel(a, b).pvalue)
                sig_rows.append({
                    "Comparison": f"{DISPLAY[prop]} vs {DISPLAY[base]}",
                    "Metric": metric,
                    "Delta": f"{delta:+.4f}",
                    "p_value": f"{p:.4g}",
                    "Significant (p<0.05)": "Yes" if p < 0.05 else "No",
                })
    if sig_rows:
        sig_table = pd.DataFrame(sig_rows)
        sig_table.to_csv(os.path.join(TAB_DIR, "significance_table.csv"), index=False)
        print("Saved table: significance_table.csv")
else:
    print("(significance table skipped -- per-seed file or scipy missing)")


# ==========================================================
# 4. RANKING TABLE  (MRR / Hits@K)  -- just copy through, tidy
# ==========================================================

if _exists("experiments/summary/ranking_metrics.csv"):
    rk = pd.read_csv(_p("experiments/summary/ranking_metrics.csv"))
    rk.to_csv(os.path.join(TAB_DIR, "ranking_table.csv"), index=False)
    print("Saved table: ranking_table.csv")


# ==========================================================
# 5. BRIDGE TABLE
# ==========================================================

bridge = None
if _exists("experiments/analysis/bridge_by_weaktie.csv"):
    bridge = pd.read_csv(_p("experiments/analysis/bridge_by_weaktie.csv"))
    bt = bridge.copy()
    bt["Model"] = bt["Model"].map(lambda x: DISPLAY.get(norm(x), x))
    bt = bt.rename(columns={"bridge": "Bridge ROC-AUC", "strong": "Overlap ROC-AUC"})
    bt.to_csv(os.path.join(TAB_DIR, "bridge_table.csv"), index=False)
    print("Saved table: bridge_table.csv")


# ==========================================================
# FIGURES
# ==========================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


# ---- Fig 1: ROC-AUC bar chart with error bars ----
roc = {}   # key -> (mean, std or 0)
for k in order:
    if k in neural and "ROC-AUC" in neural[k]:
        roc[k] = neural[k]["ROC-AUC"]
    elif k in classical and "ROC-AUC" in classical[k]:
        roc[k] = (classical[k]["ROC-AUC"], 0.0)

roc_sorted = sorted(roc.items(), key=lambda kv: kv[1][0], reverse=True)
keys = [k for k, _ in roc_sorted]
means = [v[0] for _, v in roc_sorted]
stds = [v[1] for _, v in roc_sorted]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([DISPLAY[k] for k in keys], means, yerr=stds, capsize=4,
              color=bar_colors(keys), edgecolor="black", linewidth=0.5)
ax.set_ylabel("ROC-AUC")
ax.set_title("ROC-AUC by model (mean ± std over 5 seeds; classical = single run)")
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="grey", ls="--", lw=1, label="random (0.5)")
annotate(ax, bars)
plt.xticks(rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_roc_auc.png"))
plt.close()
print("Saved figure: fig1_roc_auc.png")


# ---- Fig 2: ROC / PR / F1 grouped bars ----
grp_keys = [k for k in order if (k in neural or k in classical)]
def get_metric(k, m):
    if k in neural and m in neural[k]:
        return neural[k][m][0]
    if k in classical and m in classical[k]:
        return classical[k][m]
    return np.nan

x = np.arange(len(grp_keys))
w = 0.26
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, [get_metric(k, "ROC-AUC") for k in grp_keys], w, label="ROC-AUC", color=C_PROP)
ax.bar(x,     [get_metric(k, "PR-AUC")  for k in grp_keys], w, label="PR-AUC", color=C_PR)
ax.bar(x + w, [get_metric(k, "F1-score")for k in grp_keys], w, label="F1-score", color=C_F1)
ax.set_ylabel("Score")
ax.set_title("All classification metrics by model")
ax.set_ylim(0.5, 1.0)
ax.set_xticks(x)
ax.set_xticklabels([DISPLAY[k] for k in grp_keys], rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_all_metrics.png"))
plt.close()
print("Saved figure: fig2_all_metrics.png")


# ---- Fig 3: bridge vs strong (THE key finding) ----
if bridge is not None:
    b = bridge.copy()
    b["key"] = b["Model"].map(norm)
    b = b[b["key"].isin(order)]
    b["disp"] = b["key"].map(DISPLAY)
    b = b.sort_values("bridge", ascending=False)

    x = np.arange(len(b))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, b["bridge"].values, w, label="Bridge links", color=C_PROP, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, b["strong"].values, w, label="Overlap links", color=C_BASE, edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="random (0.5)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Bridge vs overlap links: heuristics collapse to random on bridges")
    ax.set_xticks(x)
    ax.set_xticklabels(b["disp"].values, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_bridge_vs_strong.png"))
    plt.close()
    print("Saved figure: fig3_bridge_vs_strong.png")


# ---- Fig 4: learned layer importance (beta) ----
if _exists("experiments/model_results/metrics/proposed_model_metrics.csv"):
    pm = pd.read_csv(_p("experiments/model_results/metrics/proposed_model_metrics.csv"))
    if "Layer_Importance" in pm.columns and "Layer" in pm.columns:
        pm2 = pm[["Layer", "Layer_Importance"]].sort_values("Layer_Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(pm2["Layer"], pm2["Layer_Importance"], color=C_PROP, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Learned layer importance (β)")
        ax.set_title("Cross-layer attention: which research domains matter most")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig4_layer_importance.png"))
        plt.close()
        print("Saved figure: fig4_layer_importance.png")


# ---- Fig 5: ranking (MRR + Hits@10) ----
if _exists("experiments/summary/ranking_metrics.csv"):
    rk = pd.read_csv(_p("experiments/summary/ranking_metrics.csv"))
    rk["key"] = rk["Model"].map(norm)
    rk = rk.sort_values("MRR", ascending=False)
    keys_r = rk["key"].tolist()

    x = np.arange(len(rk))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, rk["MRR"].values, w, label="MRR",
           color=bar_colors(keys_r), edgecolor="black", linewidth=0.4)
    if "Hits@10" in rk.columns:
        ax.bar(x + w/2, rk["Hits@10"].values, w, label="Hits@10",
               color=[c if c == C_PROP else "#C7CCD6" for c in bar_colors(keys_r)],
               edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.set_ylabel("Score")
    ax.set_title("Ranking quality (MRR and Hits@10)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(k, k) for k in keys_r], rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_ranking.png"))
    plt.close()
    print("Saved figure: fig5_ranking.png")


print("\nAll done.")
print("Tables :", os.path.relpath(TAB_DIR, PROJECT_ROOT))
print("Figures:", os.path.relpath(FIG_DIR, PROJECT_ROOT))
