# ==========================================================
# MULTI-SEED RUNNER  (thesis: statistical significance step)
# ==========================================================
#
# WHY THIS EXISTS
# ---------------
# Our proposed model beats the neural baselines by a small margin.
# A single run could just be luck (a lucky random weight init).
# So we run each neural model with SEVERAL random seeds and report
# mean +/- std. If the mean gap is larger than the wobble (std),
# the improvement is real -- not luck.
#
# WHAT IT DOES
# ------------
# 1. Runs the proposed model and the neural baselines once per seed, as
#    separate subprocesses (so each gets a clean, fully re-seeded process).
#    The model scripts themselves are UNCHANGED in logic -- they just
#    read THESIS_SEED / THESIS_RUN_TAG / THESIS_MAX_EPOCHS from the
#    environment, and write into experiments/multiseed/ so the
#    canonical results are never overwritten.
# 2. Reads every per-seed metrics CSV, averages the 13 layers to one
#    number per seed, then computes mean +/- std across seeds.
# 3. Runs a paired significance test (proposed vs GAT and GCN).
#
# USAGE (run from the project root)
# ---------------------------------
#   Full run (5 seeds, all 3 models):
#       python src/experiments/run_multiseed.py
#
#   Quick smoke test (fast, just to check the plumbing works):
#       python src/experiments/run_multiseed.py --smoke
#
#   Custom:
#       python src/experiments/run_multiseed.py --seeds 42 123 7 --models proposed_A gat
#
#   Only re-aggregate already-produced CSVs (no training):
#       python src/experiments/run_multiseed.py --aggregate-only
# ==========================================================

import os
import sys
import argparse
import subprocess

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

# model tag  ->  script path (relative to project root)
# All four are NEURAL models (random weight init / dropout), so multi-seed
# error bars apply to each. Classical heuristics (AA/CN/Jaccard/PA) are
# deterministic -- a single fixed number -- so they are not multi-seeded.
MODEL_SCRIPTS = {
    "proposed_A": "src/proposed_model/proposed_model.py",
    "gat":        "src/baselines/gnn_baselines/gat.py",
    "gcn":        "src/baselines/gnn_baselines/gcn.py",
    # Node2Vec is also stochastic (random walks + classifier init), so it
    # gets multi-seed error bars too. It is NOT part of the paired
    # significance test vs the proposed model, because it does not use the
    # same 6 features -- the significance claim is about architecture
    # (proposed vs gat/gcn, which share the features). Node2Vec still
    # appears in the mean +/- std table for completeness.
    "node2vec":   "src/baselines/embedding_baselines/node2vec_baseline.py",
}

DEFAULT_SEEDS = [42, 123, 2024, 7, 99]

METRICS = ["ROC-AUC", "PR-AUC", "F1-score"]

MULTISEED_METRICS_DIR = os.path.join(PROJECT_ROOT, "experiments/multiseed/metrics")
SUMMARY_DIR = os.path.join(PROJECT_ROOT, "experiments/multiseed")


# ----------------------------------------------------------
# 1. RUN one model at one seed (as a subprocess)
# ----------------------------------------------------------

def run_one(model_tag, seed, max_epochs=None):

    script = MODEL_SCRIPTS[model_tag]

    env = os.environ.copy()
    env["THESIS_SEED"] = str(seed)
    env["THESIS_RUN_TAG"] = f"seed{seed}"
    if max_epochs is not None:
        env["THESIS_MAX_EPOCHS"] = str(max_epochs)

    print(f"\n>>> Running {model_tag}  (seed={seed}"
          + (f", epochs={max_epochs}" if max_epochs else "")
          + ") ...")

    result = subprocess.run(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        env=env,
    )

    if result.returncode != 0:
        print(f"!!! {model_tag} seed {seed} FAILED "
              f"(exit code {result.returncode}). See output above.")
        return False

    return True


# ----------------------------------------------------------
# 2. AGGREGATE per-seed CSVs -> mean +/- std
# ----------------------------------------------------------

def per_seed_layer_mean(model_tag, seed):
    """Read one per-seed CSV and return the mean-over-layers for each
    metric (a single number per metric for this seed)."""

    path = os.path.join(MULTISEED_METRICS_DIR, f"{model_tag}_seed{seed}.csv")

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    out = {}
    for metric in METRICS:
        if metric in df.columns:
            out[metric] = float(df[metric].mean())
    return out


def aggregate(models, seeds):
    """Build (a) a tidy long table of per-seed scores and (b) a summary
    of mean +/- std per model/metric. Returns (long_df, summary_df,
    per_seed_scores) where per_seed_scores[model][metric] = [v_seed1, ...]."""

    long_rows = []
    per_seed_scores = {m: {metric: [] for metric in METRICS} for m in models}

    for model_tag in models:
        for seed in seeds:
            vals = per_seed_layer_mean(model_tag, seed)
            if vals is None:
                print(f"    (missing results for {model_tag} seed {seed} "
                      f"-- skipped)")
                continue
            for metric, v in vals.items():
                long_rows.append({
                    "model": model_tag,
                    "seed": seed,
                    "metric": metric,
                    "score": v,
                })
                per_seed_scores[model_tag][metric].append(v)

    long_df = pd.DataFrame(long_rows)

    summary_rows = []
    for model_tag in models:
        for metric in METRICS:
            vals = per_seed_scores[model_tag][metric]
            if len(vals) == 0:
                continue
            arr = np.array(vals, dtype=float)
            summary_rows.append({
                "model": model_tag,
                "metric": metric,
                "mean": arr.mean(),
                "std": arr.std(ddof=1) if len(arr) > 1 else 0.0,
                "n_seeds": len(arr),
                "min": arr.min(),
                "max": arr.max(),
            })

    summary_df = pd.DataFrame(summary_rows)
    return long_df, summary_df, per_seed_scores


# ----------------------------------------------------------
# 3. PAIRED significance tests (proposed vs GAT and GCN)
# ----------------------------------------------------------

def paired_test(name_a, vals_a, name_b, vals_b, metric):
    """Paired comparison on the SAME seeds. Prints the mean gap and a
    p-value. Small p (< 0.05) => the gap is unlikely to be luck."""

    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)

    n = min(len(a), len(b))
    if n < 2:
        print(f"  [{metric}] {name_a} vs {name_b}: need >=2 shared seeds "
              f"(have {n}) -- skipped")
        return

    a, b = a[:n], b[:n]
    gap = a.mean() - b.mean()

    line = (f"  [{metric}] {name_a} - {name_b} = {gap:+.4f}   "
            f"({name_a} {a.mean():.4f} vs {name_b} {b.mean():.4f}, n={n})")

    if HAVE_SCIPY:
        # paired t-test; Wilcoxon as a non-parametric backup
        try:
            t_p = stats.ttest_rel(a, b).pvalue
        except Exception:
            t_p = float("nan")
        try:
            w_p = stats.wilcoxon(a, b).pvalue if np.any(a - b != 0) else 1.0
        except Exception:
            w_p = float("nan")
        verdict = "SIGNIFICANT" if (t_p < 0.05) else "not significant"
        line += f"\n        paired t-test p = {t_p:.4f}  ({verdict});  Wilcoxon p = {w_p:.4f}"
    else:
        line += "\n        (install scipy for p-values: pip install scipy)"

    print(line)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Multi-seed runner for the thesis models.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds to run (default: 42 123 2024 7 99)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=list(MODEL_SCRIPTS.keys()),
                        help="which models (default: all three)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="cap training epochs (for quick tests)")
    parser.add_argument("--smoke", action="store_true",
                        help="fast plumbing test: 2 seeds, all models, 5 epochs")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip training, just re-read existing CSVs")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(DEFAULT_SEEDS)
    models = args.models if args.models else list(MODEL_SCRIPTS.keys())
    epochs = args.epochs

    if args.smoke:
        seeds = [42, 123]
        epochs = 5
        print("SMOKE TEST: seeds=[42, 123], epochs=5, all models "
              "(results will NOT be meaningful -- this only checks the pipeline).")

    print("=" * 60)
    print("MULTI-SEED RUN")
    print("  models:", models)
    print("  seeds :", seeds)
    print("  epochs:", epochs if epochs else "(model default)")
    print("=" * 60)

    # ---- 1. train ----
    if not args.aggregate_only:
        failures = 0
        for model_tag in models:
            for seed in seeds:
                ok = run_one(model_tag, seed, max_epochs=epochs)
                if not ok:
                    failures += 1
        if failures:
            print(f"\nWARNING: {failures} run(s) failed. "
                  f"Aggregation will use whatever completed.")

    # ---- 2. aggregate ----
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS  (mean over 13 layers, then mean +/- std over seeds)")
    print("=" * 60)

    long_df, summary_df, per_seed = aggregate(models, seeds)

    if summary_df.empty:
        print("No results found to aggregate. Did the runs complete?")
        return

    # pretty print, per metric
    for metric in METRICS:
        sub = summary_df[summary_df["metric"] == metric]
        if sub.empty:
            continue
        print(f"\n{metric}:")
        sub = sub.sort_values("mean", ascending=False)
        for _, r in sub.iterrows():
            print(f"  {r['model']:<14} {r['mean']:.4f}  +/-  {r['std']:.4f}"
                  f"   (n={int(r['n_seeds'])}, range {r['min']:.4f}-{r['max']:.4f})")

    # ---- 3. significance ----
    print("\n" + "=" * 60)
    print("SIGNIFICANCE  (paired across seeds; p < 0.05 => gap is real, not luck)")
    print("=" * 60)

    for metric in METRICS:
        have = {m: per_seed[m][metric] for m in models if per_seed[m][metric]}
        # the proposed model vs each neural baseline it should beat
        for base in ("gat", "gcn"):
            if "proposed_A" in have and base in have:
                paired_test("proposed_A", have["proposed_A"], base, have[base], metric)

    # ---- 4. save ----
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    long_path = os.path.join(SUMMARY_DIR, "multiseed_per_seed.csv")
    summary_path = os.path.join(SUMMARY_DIR, "multiseed_summary.csv")
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print("  per-seed scores :", os.path.relpath(long_path, PROJECT_ROOT))
    print("  mean +/- std    :", os.path.relpath(summary_path, PROJECT_ROOT))
    print("\nDone.")


if __name__ == "__main__":
    main()
