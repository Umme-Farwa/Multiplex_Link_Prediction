# ==========================================================
# ABLATION RUNNER
# Runs proposed_model_ablation.py for each ablation x each seed,
# reuses the existing proposed_A_seed*.csv as the "Full model",
# and prints + saves an ablation table (mean +/- std, p-values).
# proposed_model.py is NOT touched.
#
# USAGE (from project root, GPU .venv):
#   .\.venv\Scripts\python.exe src/experiments/run_ablation.py
#   .\.venv\Scripts\python.exe src/experiments/run_ablation.py --smoke
#   .\.venv\Scripts\python.exe src/experiments/run_ablation.py --aggregate-only
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ABLATION_SCRIPT = "src/proposed_model/proposed_model_ablation.py"

ABLATIONS = {
    "no_weaktie":   "abl_no_weaktie",
    "no_layerattn": "abl_no_layerattn",
    "no_fusion":    "abl_no_fusion",
}
FULL_TAG = "proposed_A"
DEFAULT_SEEDS = [42, 123, 2024, 7, 99]
METRICS = ["ROC-AUC", "PR-AUC", "F1-score"]

MULTISEED_METRICS_DIR = os.path.join(PROJECT_ROOT, "experiments/multiseed/metrics")
SUMMARY_DIR = os.path.join(PROJECT_ROOT, "experiments/ablation")


def run_one(ablation, seed, max_epochs=None):
    env = os.environ.copy()
    env["THESIS_ABLATION"] = ablation
    env["THESIS_SEED"] = str(seed)
    env["THESIS_RUN_TAG"] = f"seed{seed}"
    if max_epochs is not None:
        env["THESIS_MAX_EPOCHS"] = str(max_epochs)
    print(f"\n>>> Ablation '{ablation}' (seed={seed}"
          + (f", epochs={max_epochs}" if max_epochs else "") + ") ...")
    result = subprocess.run([sys.executable, ABLATION_SCRIPT], cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print(f"!!! {ablation} seed {seed} FAILED (exit {result.returncode}).")
        return False
    return True


def per_seed_layer_mean(tag, seed):
    path = os.path.join(MULTISEED_METRICS_DIR, f"{tag}_seed{seed}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {m: float(df[m].mean()) for m in METRICS if m in df.columns}


def collect(tag, seeds):
    scores = {m: [] for m in METRICS}
    for seed in seeds:
        vals = per_seed_layer_mean(tag, seed)
        if vals is None:
            print(f"    (missing {tag} seed {seed})")
            continue
        for m, v in vals.items():
            scores[m].append(v)
    return scores


def summarise(scores):
    out = {}
    for m in METRICS:
        arr = np.array(scores[m], dtype=float)
        if len(arr) == 0:
            out[m] = (float("nan"), float("nan"), 0)
        else:
            out[m] = (arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0, len(arr))
    return out


def paired_p(full_vals, abl_vals):
    a = np.array(full_vals, dtype=float)
    b = np.array(abl_vals, dtype=float)
    n = min(len(a), len(b))
    if n < 2 or not HAVE_SCIPY:
        return float("nan")
    try:
        return stats.ttest_rel(a[:n], b[:n]).pvalue
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(DEFAULT_SEEDS)
    epochs = args.epochs
    if args.smoke:
        seeds = [42, 123]
        epochs = 5
        print("SMOKE TEST: seeds=[42,123], epochs=5 (numbers NOT meaningful).")

    print("=" * 60)
    print("ABLATION RUN  |  ablations:", list(ABLATIONS.keys()), " seeds:", seeds,
          " epochs:", epochs if epochs else "(default)")
    print("=" * 60)

    if not args.aggregate_only:
        failures = 0
        for ablation in ABLATIONS:
            for seed in seeds:
                if not run_one(ablation, seed, max_epochs=epochs):
                    failures += 1
        if failures:
            print(f"\nWARNING: {failures} run(s) failed.")

    print("\n" + "=" * 60)
    print("ABLATION RESULTS (mean over 13 layers, then mean +/- std over seeds)")
    print("=" * 60)

    full_scores = collect(FULL_TAG, seeds)
    full_sum = summarise(full_scores)

    rows = [("Full model", full_sum, full_scores)]
    for ablation, tag in ABLATIONS.items():
        sc = collect(tag, seeds)
        rows.append((f"- {ablation}", summarise(sc), sc))

    for m in METRICS:
        print(f"\n{m}:")
        base_mean = full_sum[m][0]
        for name, summ, _ in rows:
            mean, std, n = summ[m]
            delta = "" if name == "Full model" else f"   (delta {mean - base_mean:+.4f})"
            print(f"  {name:<18} {mean:.4f} +/- {std:.4f}  (n={n}){delta}")

    print("\nSignificance of each drop vs Full model (paired t-test, ROC-AUC):")
    for name, _, sc in rows[1:]:
        p = paired_p(full_scores["ROC-AUC"], sc["ROC-AUC"])
        verdict = "SIGNIFICANT" if (p == p and p < 0.05) else "not significant / n<2"
        print(f"  Full vs {name:<18} p = {p:.4f}  ({verdict})")

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    out_rows = []
    for name, summ, _ in rows:
        row = {"variant": name.replace("- ", "without ")}
        for m in METRICS:
            mean, std, _n = summ[m]
            row[m + "_mean"] = mean
            row[m + "_std"] = std
        row["n_seeds"] = summ["ROC-AUC"][2]
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(os.path.join(SUMMARY_DIR, "ablation_summary.csv"), index=False)
    print("\nSaved:", os.path.relpath(os.path.join(SUMMARY_DIR, "ablation_summary.csv"), PROJECT_ROOT))
    print("Done.")


if __name__ == "__main__":
    main()