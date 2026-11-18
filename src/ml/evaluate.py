import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification performance metrics.

    y_true: Actual labels (0 or 1)
    y_pred: Predicted labels (0 or 1)
    y_proba: Predicted probability of class 1
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted labels must have equal lengths.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Actual labels must contain only 0 and 1.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Predicted labels must contain only 0 and 1.")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": None,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        if y_proba.ndim == 2:
            if y_proba.shape[1] != 2:
                raise ValueError("Probability array must have two columns.")
            y_proba = y_proba[:, 1]

        if len(y_proba) != len(y_true):
            raise ValueError("Probabilities and labels must have equal lengths.")

        if not np.isfinite(y_proba).all():
            raise ValueError("Probabilities must be finite.")

        if ((y_proba < 0) | (y_proba > 1)).any():
            raise ValueError("Probabilities must be between 0 and 1.")

        metrics["log_loss"] = log_loss(
            y_true,
            np.column_stack([1 - y_proba, y_proba]),
            labels=[0, 1],
        )

    return metrics


def calculate_top1_accuracy(scored):
    """
    Calculate Top-1 accuracy.

    For each GPS observation:
    1. Find the candidate with the highest predicted probability.
    2. Compare it with the verified correct candidate.
    3. Exclude groups without exactly one positive label.
    """

    required_columns = {
        "trip_id",
        "obs_idx",
        "arc_id",
        "label",
        "p_positive",
    }

    missing = required_columns - set(scored.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if scored.empty:
        raise ValueError("Cannot evaluate an empty dataset.")

    if scored[list(required_columns)].isnull().any().any():
        raise ValueError("Required columns contain missing values.")

    if not scored["label"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    probabilities = scored["p_positive"].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    groups = scored.groupby(
        ["trip_id", "obs_idx"],
        sort=False,
    )

    positive_counts = groups["label"].sum()

    valid_groups = positive_counts[positive_counts == 1].index

    # Select the highest-probability candidate per observation.
    best_indices = groups["p_positive"].idxmax()

    best_candidates = scored.loc[best_indices]

    # Keep only observations with exactly one verified positive.
    best_candidates = best_candidates.set_index(["trip_id", "obs_idx"])

    best_candidates = best_candidates.loc[valid_groups]

    correct = best_candidates["label"].to_numpy()

    total_groups = len(positive_counts)
    evaluated_groups = len(valid_groups)

    return {
        "top1_accuracy": (float(np.mean(correct)) if evaluated_groups else None),
        "total_groups": total_groups,
        "evaluated_groups": evaluated_groups,
        "excluded_groups": total_groups - evaluated_groups,
    }


def measure_inference_time(model, X, repeats=5):
    """
    Measure average prediction time for a batch of observations.

    Returns total batch prediction time, not per-row latency.
    """

    if repeats < 1:
        raise ValueError("Repeats must be at least 1.")

    if len(X) == 0:
        raise ValueError("Cannot measure inference on empty data.")

    # Warm-up prediction.
    model.predict_proba(X)

    start = time.perf_counter()

    for _ in range(repeats):
        model.predict_proba(X)

    elapsed = time.perf_counter() - start

    average_seconds = elapsed / repeats

    return {
        "average_seconds": average_seconds,
        "average_milliseconds": average_seconds * 1000,
        "repeats": repeats,
        "number_of_rows": len(X),
    }
