# Multiplex Link Prediction – Weak-Tie-Aware Cross-Layer Attention Network

This is the code for my master's thesis, *Bridging the Layers: A Weak-Tie-Aware
Cross-Layer Attention Network for Link Prediction in Multiplex Networks*.

The task is link prediction on a multiplex network: the arXiv co-authorship
network, which has 13 layers (one per research area). The proposed model is an
attention-based graph neural network that works across the layers. It is
compared against classical heuristics, Node2Vec, GCN and GAT.

The thesis write-up itself is kept on Overleaf, not in this repository.

## Folder layout

```
data/
  raw/           original dataset (edge list + layer names)
  processed/     the per-layer graphs and the train/val/test splits (.pkl)

src/
  data_processing/   load the dataset, build the graphs, split the edges
  baselines/         classical heuristics, Node2Vec, GCN, GAT
  proposed_model/    the proposed model + the ablation variants
  evaluation/        metrics (ROC-AUC, PR-AUC, F1, Hits@K, MRR) and ranking
  experiments/       multi-seed runner, ablation runner, results/figures script
  analysis/          weak-tie vs common-neighbour stratified analysis
  utils/             saving results to CSV

experiments/      all the saved results (metrics + predictions) for every method,
                  the multi-seed runs, the ablation and the analysis

report_figures/   the figures
```

The processed data is already in `data/processed/`, so the preprocessing step
below can be skipped unless you want to regenerate it.

## Requirements

Python 3.11. Packages: torch, torch-geometric, networkx, scikit-learn,
node2vec, numpy, pandas, matplotlib, scipy.

## How to run

All commands are run from the project root.

1. (optional) rebuild the data:
   ```
   python src/data_processing/preprocess.py
   python src/data_processing/split_edges.py
   ```

2. baselines:
   ```
   python src/baselines/classical_baselines/adamic_adar.py
   python src/baselines/classical_baselines/common_neighbors.py
   python src/baselines/classical_baselines/jaccard.py
   python src/baselines/classical_baselines/preferential_attachment.py
   python src/baselines/embedding_baselines/node2vec_baseline.py
   python src/baselines/gnn_baselines/gcn.py
   python src/baselines/gnn_baselines/gat.py
   ```

3. the proposed model:
   ```
   python src/proposed_model/proposed_model.py
   ```

4. multi-seed runs (needed for the mean ± std and the significance test):
   ```
   python src/experiments/run_multiseed.py
   ```

5. ablation (removes one component at a time; uses the multi-seed runs):
   ```
   python src/experiments/run_ablation.py
   ```

6. analysis and the final tables/figures:
   ```
   python src/analysis/bridge_link_analysis.py
   python src/evaluation/ranking_eval.py
   python src/experiments/make_results.py
   ```

Metrics and predictions are written as CSV files under `experiments/`, and the
figures go to `report_figures/`.

## Note on the splits

All methods use the same seed (42) for the train/val/test split, so they are all
trained and tested on exactly the same data. The F1 threshold is tuned only on
the validation set, never on the test set.

## Note on naming

In the multi-seed scripts and the raw result files the proposed model is tagged
`proposed_A`. That is just the internal name for the proposed model; in the
thesis and the summary tables it is simply called "Proposed".
