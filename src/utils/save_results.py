import os
import pandas as pd


def save_metrics(results, filename):
    """
    Save evaluation metrics.
    """

    folder = "experiments/baseline_results/metrics"

    os.makedirs(
        folder,
        exist_ok=True
    )

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(folder, filename),
        index=False
    )


def save_predictions(predictions, filename):
    """
    Save prediction scores.
    """

    folder = "experiments/baseline_results/predictions"

    os.makedirs(
        folder,
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
        os.path.join(folder, filename),
        index=False
    )