
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.features import FEATURE_COLUMNS

DATA_DIR = Path("data/processed/train")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("results/ml")

RANDOM_STATE = 42


def load_training_data():
    frames = []

    for path in DATA_DIR.glob("*.parquet"):
        if path.stem.isdigit() and len(path.stem) == 8:
            df = pd.read_parquet(path)
            df = df[df["training_eligible"] == True].copy()
            df = df[df["label"].notna()].copy()
            df["trip_id"] = path.stem
            frames.append(df)

    if not frames:
        raise RuntimeError("No eligible training data found.")

    return pd.concat(frames, ignore_index=True)


def train_final_model():
    print("Loading all training trips...", flush=True)

    data = load_training_data()

    X = data[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    y = data["label"].astype(int)

    print(f"Training rows: {len(data):,}", flush=True)
    print(f"Training trips: {data['trip_id'].nunique()}", flush=True)

    params_path = RESULTS_DIR / "best_params.json"

    with open(params_path, "r") as file:
        tuning_results = json.load(file)

    best_params = tuning_results["best_parameters"]

    print("Using Optuna's best parameters:", best_params, flush=True)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        ))
    ])

    print("\nTraining final XGBoost model...", flush=True)
    model.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "final_pipeline.joblib"
    metadata_path = ARTIFACTS_DIR / "model_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "project": "PathSnap",
        "model_name": "XGBoost",
        "features": FEATURE_COLUMNS,
        "positive_class": 1,
        "positive_class_meaning": "Correct road candidate",
        "negative_class": 0,
        "negative_class_meaning": "Incorrect road candidate",
        "training_trips": int(data["trip_id"].nunique()),
        "training_rows": int(len(data)),
        "hyperparameters": best_params,
        "optuna_best_macro_f1": tuning_results["best_macro_f1"],
        "random_state": RANDOM_STATE,
        "prediction_output": "predict_proba",
    }

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nFINAL MODEL EXPORTED SUCCESSFULLY!")
    print("Model:", model_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    train_final_model()
