import os
import pandas as pd


def save_metrics(results, filename, base_folder="experiments/baseline_results/metrics"):
    """
    Save evaluation metrics.

    base_folder:
        Root directory to save under. Defaults to the baseline metrics
        folder, but callers (e.g. the proposed model or ablation runs)
        can point it elsewhere, e.g. "experiments/model_results/metrics".
    filename:
        Path relative to base_folder, may include subfolders
        (e.g. "classical/adamic_adar_metrics.csv").
    """

    path = os.path.join(base_folder, filename)

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df = pd.DataFrame(results)

    df.to_csv(
        path,
        index=False
    )


def save_predictions(predictions, filename, base_folder="experiments/baseline_results/predictions"):
    """
    Save prediction scores.

    base_folder:
        Root directory to save under. Defaults to the baseline
        predictions folder; callers can override it (e.g. the proposed
        model uses "experiments/model_results/predictions").
    filename:
        Path relative to base_folder, may include subfolders
        (e.g. "classical/adamic_adar_predictions.csv").
    """

    path = os.path.join(base_folder, filename)

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df = pd.DataFrame(
        predictions,
        columns=[
            "node1",
            "node2",
            "score",
            "label",
            "layer"
        ]
    )

    df.to_csv(
        path,
        index=False
    )