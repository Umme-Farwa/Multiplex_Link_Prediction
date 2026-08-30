import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve
)


def classification_metrics(labels, scores, threshold=0.5):
    """
    Calculate ROC-AUC, PR-AUC and F1-score.

    labels:
        True labels (1 = link, 0 = no link)

    scores:
        Predicted link probabilities

    threshold:
        Cutoff used to convert scores into binary predictions for F1.
        Default 0.5, but callers can pass a threshold tuned on a
        validation set (see find_best_threshold) for fairer comparison
        across models whose scores may not be equally calibrated.
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


def find_best_threshold(labels, scores):
    """
    Find the classification threshold that maximizes F1-score.

    IMPORTANT: this should be called on VALIDATION labels/scores only
    (never on the test set). Using the test set to pick the threshold
    would leak test information into the evaluation and inflate the
    reported F1-score.

    Uses precision_recall_curve to get all candidate thresholds
    efficiently (rather than scanning every unique score value),
    then computes F1 from precision/recall directly.

    Returns:
        best_threshold (float), best_f1 (float)
    """

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # precision/recall have one more element than thresholds
    # (the last point corresponds to threshold = +inf, recall = 0),
    # so we drop that last point before matching against thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall

    f1_scores = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom != 0
    )

    if len(f1_scores) == 0:
        return 0.5, 0.0

    best_idx = int(np.argmax(f1_scores))

    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_f1


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


def evaluate_all(labels, scores, threshold=0.5):
    """
    Complete evaluation for binary link prediction.

    threshold:
        Optional cutoff for F1 (default 0.5). Pass a
        validation-tuned threshold (from find_best_threshold) for
        fairer F1 comparison across models with different score
        calibration.
    """

    results = classification_metrics(
        labels,
        scores,
        threshold=threshold
    )

    return results