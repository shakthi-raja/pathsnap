"""Compute the quality, missing-value and per-model measurements the report needs.

Produces three files that the report and the figure scripts read:

* ``results/data_quality.json``: sampling intervals, label coverage and the
  count of observations the audit looked at.
* ``results/missing_values.csv``: missing values per column, per partition.
* ``results/model_test_metrics.csv``: precision, recall, macro F1, log loss,
  Top-1 accuracy, candidate-level accuracy, classifier-path route continuity
  and inference time for every model on the student-test partition.

The training frames and the fitted pipelines are identical to
:mod:`src.ml.train_models`, so the numbers here line up with the comparison
table that script already writes.

Usage::

    PYTHONPATH=. python tools/report_metrics.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from src.api.evaluation import route_continuity
from src.config import get_settings
from src.data.features import FEATURE_COLUMNS
from src.data.network import build_road_network
from src.data.reader import load_trip
from src.ml.dataset import list_trip_ids, model_frame
from src.ml.evaluate import calculate_top1_accuracy
from src.ml.train_models import build_models
from src.routing.transitions import NetworkTravel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
RAW = get_settings().data_root
PROCESSED = RAW.parent / "processed"


def _non_finite(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce")
    return int((~np.isfinite(values)).sum())


def data_quality() -> dict[str, object]:
    """Sampling intervals, label coverage and dataset size."""
    buckets = {"gap_1s": 0, "gap_2_to_5s": 0, "gap_6_to_60s": 0, "gap_over_60s": 0}
    trips = 0
    observations = 0
    non_finite = 0
    for track_path in sorted((RAW / "train").glob("*/*.track")):
        values = np.loadtxt(track_path, usecols=(0, 1, 2), ndmin=2)
        if values.size == 0:
            continue
        trips += 1
        observations += len(values)
        non_finite += int((~np.isfinite(values)).sum())
        if len(values) < 2:
            continue
        stamps = np.sort(values[:, 2])
        gaps = np.diff(stamps)
        buckets["gap_1s"] += int((gaps <= 1).sum())
        buckets["gap_2_to_5s"] += int(((gaps > 1) & (gaps <= 5)).sum())
        buckets["gap_6_to_60s"] += int(((gaps > 5) & (gaps <= 60)).sum())
        buckets["gap_over_60s"] += int((gaps > 60).sum())

    summary = pd.read_csv(PROCESSED / "train" / "processing_summary.csv")
    confident = int(summary["confident_observations"].sum())
    uncertain = int(summary["uncertain_observations"].sum())

    candidate_rows = 0
    labelled_candidate_rows = 0
    labelled_trips = 0
    training_trip_ids = list_trip_ids(PROCESSED / "train")
    for trip_id in training_trip_ids:
        frame = pd.read_parquet(PROCESSED / "train" / f"{trip_id}.parquet")
        candidate_rows += len(frame)
        eligible = frame["training_eligible"].fillna(False).astype(bool)
        trip_labelled = int((eligible & frame["label"].notna()).sum())
        labelled_candidate_rows += trip_labelled
        labelled_trips += int(trip_labelled > 0)

    gaps_total = sum(buckets.values())
    return {
        "training_trips": trips,
        "observations": observations,
        "raw_non_finite_values": non_finite,
        "gaps": {name: {"count": value, "share": value / gaps_total} for name, value in buckets.items()},
        "labelled_observations": confident,
        "withheld_observations": uncertain,
        "labelled_share": confident / (confident + uncertain),
        "candidate_rows": candidate_rows,
        "labelled_candidate_rows": labelled_candidate_rows,
        "labelled_candidate_share": labelled_candidate_rows / candidate_rows,
        "labelled_trips": labelled_trips,
        "training_trip_tables": len(training_trip_ids),
    }


def missing_values() -> pd.DataFrame:
    """Missing values per column for both labelled partitions."""
    rows = []
    for partition in ("train", "student_test"):
        directory = PROCESSED / partition
        total = 0
        counts: dict[str, int] = {}
        for trip_id in list_trip_ids(directory):
            frame = pd.read_parquet(directory / f"{trip_id}.parquet")
            total += len(frame)
            for column in frame.columns:
                counts[column] = counts.get(column, 0) + _non_finite(frame[column])
        for column, missing in counts.items():
            rows.append(
                {
                    "partition": partition,
                    "column": column,
                    "missing": missing,
                    "rows": total,
                    "share": missing / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _positive_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return probabilities[:, list(model.classes_).index(1)]


def _classifier_continuity(
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    travel: NetworkTravel,
) -> float | None:
    """Continuity of the highest scoring candidate per observation."""
    scored = candidates[["gps_index", "arc_id", "from_node_id", "to_node_id"]].copy()
    scored["p_positive"] = probabilities
    best = (
        scored.sort_values(["gps_index", "p_positive"], ascending=[True, False])
        .drop_duplicates("gps_index")
        .sort_values("gps_index")
        .rename(columns={"gps_index": "obs_idx"})
    )
    return route_continuity(best, travel)


def model_test_metrics() -> pd.DataFrame:
    """Fit each model on the full training partition and score student-test."""
    train = model_frame(PROCESSED / "train")
    test = model_frame(PROCESSED / "student_test")
    train_features = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    test_features = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # Collect the student-test candidate tables and networks once, so the
    # continuity measurement does not reload them per model.
    test_tables = {}
    test_travels = {}
    for trip_id in list_trip_ids(PROCESSED / "student_test"):
        table = pd.read_parquet(PROCESSED / "student_test" / f"{trip_id}.parquet")
        test_tables[trip_id] = table
        _, nodes, arcs, _ = load_trip(RAW / "student_test", trip_id)
        network = build_road_network(nodes, arcs)
        test_travels[trip_id] = NetworkTravel(network.adjacency)

    rows = []
    for name, pipeline in build_models().items():
        started = time.perf_counter()
        pipeline.fit(train_features, train["label"])
        fit_seconds = time.perf_counter() - started

        timings = []
        for _ in range(3):
            started = time.perf_counter()
            probabilities = _positive_probabilities(pipeline, test_features)
            timings.append((time.perf_counter() - started) * 1000.0)
        inference_ms = min(timings)
        predictions = pipeline.predict(test_features)

        scored = test[["trip_id", "obs_idx", "arc_id", "label"]].copy()
        scored["p_positive"] = probabilities
        top1 = calculate_top1_accuracy(scored)["top1_accuracy"]

        continuity = []
        for trip_id, table in test_tables.items():
            trip_probabilities = _positive_probabilities(
                pipeline, table[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
            )
            value = _classifier_continuity(table, trip_probabilities, test_travels[trip_id])
            if value is not None:
                continuity.append(value)

        rows.append(
            {
                "model": name,
                "test_accuracy": float(accuracy_score(test["label"], predictions)),
                "test_precision": float(precision_score(test["label"], predictions, zero_division=0)),
                "test_recall": float(recall_score(test["label"], predictions, zero_division=0)),
                "test_macro_f1": float(f1_score(test["label"], predictions, average="macro", zero_division=0)),
                "test_log_loss": float(log_loss(test["label"], pipeline.predict_proba(test_features))),
                "test_top1": float(top1),
                "classifier_continuity": float(np.mean(continuity)) if continuity else None,
                "fit_seconds": fit_seconds,
                "inference_ms": inference_ms,
            }
        )
        print(
            f"  {name}: precision {rows[-1]['test_precision']:.3f} "
            f"recall {rows[-1]['test_recall']:.3f} continuity "
            f"{rows[-1]['classifier_continuity']:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Data quality...", flush=True)
    (RESULTS / "data_quality.json").write_text(
        json.dumps(data_quality(), indent=2), encoding="utf-8"
    )

    print("Missing values...", flush=True)
    missing_values().to_csv(RESULTS / "missing_values.csv", index=False)

    print("Per-model student-test metrics...", flush=True)
    model_test_metrics().to_csv(RESULTS / "model_test_metrics.csv", index=False)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
