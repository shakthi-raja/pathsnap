"""Train and compare candidate classifiers, then export the chosen pipeline.

Five families are compared with grouped cross-validation over trips, which is
also cross-city validation here: every trip in the dataset sits in a different
city. Selection uses observation-level Top-1 rather than row accuracy, because
most candidate rows are negatives and row accuracy flatters a lazy model.

Usage::

    python -m src.ml.train_models
    python -m src.ml.train_models --max-trips 8      # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import get_settings
from src.data.features import FEATURE_COLUMNS
from src.ml.dataset import model_frame
from src.ml.evaluate import calculate_top1_accuracy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The dataset lives outside the repository, so processed tables sit beside the
# raw partitions rather than inside the working tree.
PROCESSED_DIR = get_settings().data_root.parent / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
N_SPLITS = 5


def build_models() -> dict[str, object]:
    """Return the five candidates, one per algorithm family.

    Every preprocessing step lives inside the pipeline so it is fitted on the
    training fold only. Fitting an imputer on the full dataset before splitting
    is one of the quietest ways to leak validation information.
    """
    return {
        "GaussianNB": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GaussianNB(),
        ),
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "DecisionTree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=18, random_state=RANDOM_STATE),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=80,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def _positive_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class, whatever column sklearn chose."""
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def _top1(model, frame: pd.DataFrame, features: pd.DataFrame) -> dict[str, float | None]:
    """Observation-level Top-1 for one fitted model."""
    scored = frame[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    scored["p_positive"] = _positive_scores(model, features)
    return calculate_top1_accuracy(scored)


def _fold_metrics(
    model, train_frame: pd.DataFrame, validation_frame: pd.DataFrame
) -> dict[str, float | None]:
    """Score one fitted model on one held-out fold."""
    features = validation_frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    target = validation_frame["label"].to_numpy()
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)

    result = {
        "macro_f1": float(f1_score(target, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(target, probabilities, labels=list(model.classes_))),
        "accuracy": float((predictions == target).mean()),
    }
    result.update(_top1(model, validation_frame, features))
    return result


def cross_validate(train: pd.DataFrame, models: dict[str, object]) -> pd.DataFrame:
    """Run grouped cross-validation and return one row per model and fold."""
    target = train["label"]
    groups = train["trip_id"]

    splitter = GroupKFold(n_splits=N_SPLITS)
    rows: list[dict[str, object]] = []

    for name, model in models.items():
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(train, target, groups), start=1
        ):
            fit_frame = train.iloc[train_idx]
            validation_frame = train.iloc[valid_idx]
            started = time.perf_counter()
            model.fit(
                fit_frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan),
                fit_frame["label"],
            )
            elapsed = time.perf_counter() - started

            metrics = _fold_metrics(model, fit_frame, validation_frame)
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "train_seconds": round(elapsed, 2),
                    "train_rows": len(train_idx),
                    "validation_rows": len(valid_idx),
                    "train_trips": groups.iloc[train_idx].nunique(),
                    "validation_trips": groups.iloc[valid_idx].nunique(),
                    **metrics,
                }
            )
            print(
                f"  {name:<22} fold {fold}  macro F1 {metrics['macro_f1']:.4f}  "
                f"top-1 {metrics['top1_accuracy']:.4f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def evaluate_student_test(
    models: dict[str, object], train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Fit each model on the whole training partition and score student-test.

    The student-test partition is used for comparison only. Nothing here feeds
    back into model choice or hyperparameters.
    """
    rows: list[dict[str, object]] = []
    train_features = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    for name, model in models.items():
        started = time.perf_counter()
        model.fit(train_features, train["label"])
        fit_seconds = time.perf_counter() - started

        inference_started = time.perf_counter()
        metrics = _fold_metrics(model, train, test)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        rows.append(
            {
                "model": name,
                "student_test_rows": len(test),
                "fit_seconds": round(fit_seconds, 2),
                "inference_ms": round(inference_ms, 1),
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
        print(
            f"  {name:<22} test macro F1 {metrics['macro_f1']:.4f}  "
            f"top-1 {metrics['top1_accuracy']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame, student_test: pd.DataFrame) -> pd.DataFrame:
    """Collapse the fold table and join the student-test scores."""
    aggregated = (
        folds.groupby("model")
        .agg(
            cv_macro_f1_mean=("macro_f1", "mean"),
            cv_macro_f1_std=("macro_f1", "std"),
            cv_log_loss_mean=("log_loss", "mean"),
            cv_accuracy_mean=("accuracy", "mean"),
            cv_top1_mean=("top1_accuracy", "mean"),
            cv_top1_std=("top1_accuracy", "std"),
            cv_train_seconds_mean=("train_seconds", "mean"),
        )
        .reset_index()
    )
    aggregated = aggregated.merge(
        student_test[
            [
                "model",
                "test_macro_f1",
                "test_log_loss",
                "test_accuracy",
                "test_top1_accuracy",
                "inference_ms",
            ]
        ],
        on="model",
        how="left",
    )
    return aggregated.sort_values("cv_top1_mean", ascending=False).reset_index(drop=True)


#: Two models within this much cross-validated Top-1 are treated as tied.
TIE_TOLERANCE = 0.01


def select_best(comparison: pd.DataFrame) -> str:
    """Pick by cross-validated Top-1, with log loss as the tie-breaker.

    Top-1 matches the task: one road per observation. Models within
    ``TIE_TOLERANCE`` of the best are treated as equivalent on accuracy and the
    one with the lower log loss wins, because the decoder consumes
    probabilities rather than hard decisions.
    """
    best_top1 = comparison["cv_top1_mean"].max()
    contenders = comparison[comparison["cv_top1_mean"] >= best_top1 - TIE_TOLERANCE]
    ranked = contenders.sort_values("cv_log_loss_mean", ascending=True)
    return str(ranked.iloc[0]["model"])


def export_model(model, name: str, comparison: pd.DataFrame) -> None:
    """Persist the chosen pipeline and the metadata the API needs."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "final_pipeline.joblib")

    row = comparison.loc[comparison["model"] == name].iloc[0]
    metadata = {
        "model_name": name,
        "features": FEATURE_COLUMNS,
        "positive_class": 1,
        "description": "candidate arc is the reference match",
        "training_partition": "80 training trips",
        "selection_metric": "cross-validated observation-level Top-1",
        "random_seed": RANDOM_STATE,
        "cv_n_splits": N_SPLITS,
        "cv_top1_mean": float(row["cv_top1_mean"]),
        "cv_macro_f1_mean": float(row["cv_macro_f1_mean"]),
        "test_top1_accuracy": float(row["test_top1_accuracy"]),
        "test_macro_f1": float(row["test_macro_f1"]),
        "test_log_loss": float(row["test_log_loss"]),
    }
    (ARTIFACTS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nExported {name} to {ARTIFACTS_DIR}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-trips", type=int, default=None, help="Limit training trips for a smoke test"
    )
    args = parser.parse_args(argv)

    print("Loading candidate tables...", flush=True)
    train = model_frame(PROCESSED_DIR / "train")
    test = model_frame(PROCESSED_DIR / "student_test")

    if args.max_trips:
        keep = train["trip_id"].unique()[: args.max_trips]
        train = train.loc[train["trip_id"].isin(keep)]

    print(
        f"Train: {len(train):,} rows across {train['trip_id'].nunique()} trips\n"
        f"Student test: {len(test):,} rows across {test['trip_id'].nunique()} trips\n"
        f"Positive rate: {train['label'].mean():.3f}",
        flush=True,
    )

    models = build_models()

    print("\nCross-validating...", flush=True)
    folds = cross_validate(train, models)

    print("\nEvaluating on student test...", flush=True)
    student_test = evaluate_student_test(models, train, test)

    comparison = summarise(folds, student_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(RESULTS_DIR / "cv_fold_results.csv", index=False)
    student_test.to_csv(RESULTS_DIR / "student_test_results.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    chosen = select_best(comparison)
    print(f"\nSelected: {chosen}")
    export_model(models[chosen], chosen, comparison)


if __name__ == "__main__":
    main()
