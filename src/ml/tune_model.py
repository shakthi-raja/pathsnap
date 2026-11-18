"""Bayesian hyperparameter search for the chosen candidate classifier.

The search is grouped by trip, so every fold validates on cities the model was
not fitted on. Trials, the objective and the best configuration are all kept so
the tuned and untuned results can be compared on identical data.

Usage::

    python -m src.ml.tune_model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from src.config import get_settings
from src.data.features import FEATURE_COLUMNS
from src.ml.dataset import model_frame
from src.ml.evaluate import calculate_top1_accuracy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = get_settings().data_root.parent / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
N_SPLITS = 3
N_TRIALS = 10
OPTUNA_SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _top1(model, frame: pd.DataFrame, features: pd.DataFrame) -> float | None:
    scored = frame[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    probabilities = model.predict_proba(features)
    scored["p_positive"] = probabilities[:, list(model.classes_).index(1)]
    return calculate_top1_accuracy(scored)["top1_accuracy"]


def build_objective(train: pd.DataFrame):
    """Return the Optuna objective, closing over the training data."""
    features = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    target = train["label"]
    groups = train["trip_id"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }

        scores: list[float] = []
        for train_idx, valid_idx in GroupKFold(n_splits=N_SPLITS).split(features, target, groups):
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(**params, random_state=RANDOM_STATE),
            )
            fit_frame = train.iloc[train_idx]
            validation = train.iloc[valid_idx]
            model.fit(
                fit_frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan),
                fit_frame["label"],
            )
            validation_features = validation[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
            top1 = _top1(model, validation, validation_features)
            macro = f1_score(
                validation["label"],
                model.predict(validation_features),
                average="macro",
                zero_division=0,
            )
            scores.append(0.7 * (top1 or 0.0) + 0.3 * macro)
        return float(np.mean(scores))

    return objective


def main() -> None:
    train = model_frame(PROCESSED_DIR / "train")
    test = model_frame(PROCESSED_DIR / "student_test")
    print(f"Tuning on {len(train):,} rows across {train['trip_id'].nunique()} trips", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    study.optimize(build_objective(train), n_trials=N_TRIALS, show_progress_bar=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)

    best = study.best_params
    (RESULTS_DIR / "best_params.json").write_text(
        json.dumps({"params": best, "objective": study.best_value}, indent=2), encoding="utf-8"
    )
    print(f"Best objective {study.best_value:.4f} with {best}", flush=True)

    tuned = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(**best, random_state=RANDOM_STATE),
    )
    tuned.fit(
        train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan),
        train["label"],
    )
    test_features = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    tuned_top1 = _top1(tuned, test, test_features)

    import joblib

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned, ARTIFACTS_DIR / "tuned_pipeline.joblib")
    (ARTIFACTS_DIR / "tuned_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "HistGradientBoosting (tuned)",
                "features": FEATURE_COLUMNS,
                "positive_class": 1,
                "best_params": best,
                "objective": study.best_value,
                "n_trials": N_TRIALS,
                "cv_n_splits": N_SPLITS,
                "test_top1_accuracy": tuned_top1,
                "sampler": "TPE",
                "seed": OPTUNA_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Tuned student-test Top-1: {tuned_top1:.4f}", flush=True)


if __name__ == "__main__":
    main()
