
import pandas as pd
import pytest

from src.ml.evaluate import (
    calculate_classification_metrics,
    calculate_top1_accuracy,
)


def test_classification_metrics():
    actual = [1, 0, 1, 0]
    predicted = [1, 0, 0, 0]
    probabilities = [0.9, 0.1, 0.4, 0.2]

    results = calculate_classification_metrics(
        actual,
        predicted,
        probabilities,
    )

    assert results["accuracy"] == 0.75
    assert results["macro_f1"] > 0
    assert results["log_loss"] is not None


def test_top1_accuracy():
    data = pd.DataFrame({
        "trip_id": ["T1", "T1", "T1", "T1"],
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [10, 11, 20, 21],
        "label": [1, 0, 0, 1],
        "p_positive": [0.8, 0.2, 0.3, 0.7],
    })

    results = calculate_top1_accuracy(data)

    assert results["top1_accuracy"] == 1.0
    assert results["total_groups"] == 2
    assert results["evaluated_groups"] == 2
    assert results["excluded_groups"] == 0


def test_invalid_labels():
    with pytest.raises(ValueError):
        calculate_classification_metrics(
            [1, 2, 0],
            [1, 0, 0],
        )

