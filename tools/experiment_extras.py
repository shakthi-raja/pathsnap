"""Compute the measurements the report was missing.

Produces three files:

* ``results/tuned_vs_untuned.csv``: the five comparison metrics the guidelines
  ask for, on identical folds and identical student-test data.
* ``results/sequence_metrics.csv``: route-level transition precision, recall
  and F1 for the classifier-only path and the decoded path.
* ``results/confusion_matrix.json``: candidate-level counts for the shipped
  model on the student-test partition.

Usage::

    PYTHONPATH=. python tools/experiment_extras.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from src.config import get_settings
from src.data.features import FEATURE_COLUMNS
from src.data.labels import align_reference_route
from src.data.network import build_road_network
from src.data.reader import load_trip
from src.ml.dataset import model_frame
from src.ml.evaluate import calculate_top1_accuracy
from src.routing.decoder import decode_candidates
from src.routing.transitions import NetworkTravel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
ARTIFACTS = PROJECT_ROOT / "artifacts"
PROCESSED = get_settings().data_root.parent / "processed"
RAW = get_settings().data_root

N_SPLITS = 5


def _score(model, frame: pd.DataFrame) -> np.ndarray:
    features = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    probabilities = model.predict_proba(features)
    return probabilities[:, list(model.classes_).index(1)]


def _top1(frame: pd.DataFrame, probabilities: np.ndarray) -> float | None:
    scored = frame[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    scored["p_positive"] = probabilities
    return calculate_top1_accuracy(scored)["top1_accuracy"]


def tuned_comparison() -> pd.DataFrame:
    """Compare the untuned and tuned pipelines on the same folds and test set."""
    train = model_frame(PROCESSED / "train")
    test = model_frame(PROCESSED / "student_test")

    variants = {
        "untuned (HistGradientBoosting)": joblib.load(ARTIFACTS / "final_pipeline.joblib"),
        "tuned (Optuna TPE)": joblib.load(ARTIFACTS / "tuned_pipeline.joblib"),
    }

    rows = []
    for name, model in variants.items():
        fold_scores = []
        groups = train["trip_id"]
        splitter = GroupKFold(n_splits=N_SPLITS)
        features_all = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        for train_idx, valid_idx in splitter.split(features_all, train["label"], groups):
            model.fit(features_all.iloc[train_idx], train["label"].iloc[train_idx])
            predictions = model.predict(features_all.iloc[valid_idx])
            fold_scores.append(
                f1_score(train["label"].iloc[valid_idx], predictions, average="macro", zero_division=0)
            )

        model.fit(features_all, train["label"])
        test_features = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            probabilities = model.predict_proba(test_features)
            timings.append((time.perf_counter() - started) * 1000.0)
        inference_ms = min(timings)
        predictions = model.predict(test_features)
        positive = probabilities[:, list(model.classes_).index(1)]

        rows.append(
            {
                "variant": name,
                "cv_macro_f1": float(np.mean(fold_scores)),
                "cv_macro_f1_std": float(np.std(fold_scores)),
                "test_macro_f1": float(
                    f1_score(test["label"], predictions, average="macro", zero_division=0)
                ),
                "test_top1": _top1(test, positive),
                "test_log_loss": float(log_loss(test["label"], probabilities)),
                "inference_ms": inference_ms,
            }
        )
        print(f"  {name}: cv F1 {rows[-1]['cv_macro_f1']:.3f} top1 {rows[-1]['test_top1']:.3f}", flush=True)
    return pd.DataFrame(rows)


def _transitions(sequence: list[int]) -> set[tuple[int, int]]:
    return set(zip(sequence[:-1], sequence[1:], strict=True))


def sequence_metrics() -> pd.DataFrame:
    """Route-level transition precision, recall and F1 on student-test trips."""
    model = joblib.load(ARTIFACTS / "final_pipeline.joblib")
    rows = []

    for trip_id in sorted(p.stem for p in (PROCESSED / "student_test").glob("*.parquet")
                          if p.stem.isdigit()):
        candidates = pd.read_parquet(PROCESSED / "student_test" / f"{trip_id}.parquet")
        eligible = candidates["training_eligible"].fillna(False).astype(bool)
        if not eligible.any():
            continue

        track, nodes, arcs, route = load_trip(RAW / "student_test", trip_id)
        network = build_road_network(nodes, arcs)
        alignment = align_reference_route(track, network, route)
        reference = (
            alignment.loc[alignment["alignment_confident"], ["gps_index", "aligned_arc_id"]]
            .sort_values("gps_index")
        )
        if reference.empty:
            continue

        with np.errstate(all="ignore"):
            probabilities = _score(model, candidates.rename(columns={"gps_index": "obs_idx"}))
        travel = NetworkTravel(network.adjacency)
        decoded = decode_candidates(candidates, probabilities, travel=travel)

        reference_sequence = (
            reference.drop_duplicates("aligned_arc_id")["aligned_arc_id"].astype(int).tolist()
        )
        reference_pairs = _transitions(reference_sequence)

        for label, frame in (("classifier", decoded.classifier_only), ("hmm", decoded.decoded)):
            sequence = (
                frame.sort_values("gps_index")
                .drop_duplicates("arc_id")["arc_id"]
                .astype(int)
                .tolist()
            )
            predicted_pairs = _transitions(sequence)
            overlap = len(predicted_pairs & reference_pairs)
            precision = overlap / len(predicted_pairs) if predicted_pairs else 0.0
            recall = overlap / len(reference_pairs) if reference_pairs else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "trip_id": trip_id,
                    "path": label,
                    "predicted_arcs": len(sequence),
                    "reference_arcs": len(reference_sequence),
                    "transition_precision": precision,
                    "transition_recall": recall,
                    "transition_f1": f1,
                }
            )
        print(f"  {trip_id}: done", flush=True)
    return pd.DataFrame(rows)


def confusion() -> dict[str, int]:
    """Candidate-level confusion counts for the shipped model."""
    test = model_frame(PROCESSED / "student_test")
    model = joblib.load(ARTIFACTS / "final_pipeline.joblib")
    features = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    predictions = model.predict(features)
    matrix = confusion_matrix(test["label"], predictions, labels=[0, 1])
    return {
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Tuned versus untuned...", flush=True)
    tuned_comparison().to_csv(RESULTS / "tuned_vs_untuned.csv", index=False)

    print("Confusion matrix...", flush=True)
    (RESULTS / "confusion_matrix.json").write_text(json.dumps(confusion(), indent=2), encoding="utf-8")

    print("Sequence metrics...", flush=True)
    sequence_metrics().to_csv(RESULTS / "sequence_metrics.csv", index=False)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
