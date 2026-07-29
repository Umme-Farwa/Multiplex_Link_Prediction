import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)


def classification_metrics(labels, scores, threshold=0.5):
    """
    Calculate ROC-AUC, PR-AUC and F1-score.

    labels:
        True labels (1 = link, 0 = no link)

    scores:
        Predicted link probabilities
    """

    predictions = [
        1 if score >= threshold else 0
        for score in scores
    ]

    roc_auc = roc_auc_score(
        labels,
        scores
    )

    pr_auc = average_precision_score(
        labels,
        scores
    )

    f1 = f1_score(
        labels,
        predictions
    )

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "F1-score": f1
    }



def hits_at_k(labels, scores, k=10):
    """
    Calculate Hits@K.

    Checks how many true links
    appear in top K predictions.
    """

    ranked_indices = np.argsort(scores)[::-1]

    top_k_indices = ranked_indices[:k]

    hits = sum(
        labels[i] == 1
        for i in top_k_indices
    )

    return hits / k



def mean_reciprocal_rank(labels, scores):
    """
    Calculate Mean Reciprocal Rank (MRR).
    """

    ranked_indices = np.argsort(scores)[::-1]

    for rank, index in enumerate(ranked_indices, start=1):

        if labels[index] == 1:
            return 1 / rank

    return 0.0



def evaluate_all(labels, scores):
    """
    Complete evaluation.
    """

    results = classification_metrics(
        labels,
        scores
    )

    results["Hits@10"] = hits_at_k(
        labels,
        scores,
        k=10
    )

    results["MRR"] = mean_reciprocal_rank(
        labels,
        scores
    )

    return results  