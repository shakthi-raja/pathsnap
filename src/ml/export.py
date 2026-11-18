
import json
from pathlib import Path

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.ml.train_models import (
    FEATURES,
    RANDOM_STATE,
    PROJECT_ROOT,
    load_training_data,
)


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def export_model(model, model_name, metadata=None):

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = ARTIFACTS_DIR / "final_pipeline.joblib"

    metadata_path = ARTIFACTS_DIR / "model_metadata.json"

    # Save the complete pipeline.
    joblib.dump(model, model_path)

    model_metadata = {
        "project": "PathSnap",
        "model_name": model_name,
        "features": FEATURES,
        "positive_class": 1,
        "positive_class_meaning": "Correct road candidate",
        "negative_class": 0,
        "negative_class_meaning": "Incorrect road candidate",
        "random_state": RANDOM_STATE,
        "training_partition": "80 training trips",
    }

    if metadata:
        model_metadata.update(metadata)

    with open(metadata_path, "w") as file:
        json.dump(
            model_metadata,
            file,
            indent=4,
        )

    print("\nModel exported successfully!")
    print("Model:", model_path)
    print("Metadata:", metadata_path)


def train_final_model(best_params=None):

    data = load_training_data()

    X = data[FEATURES].copy()
    y = data["label"].astype(int)

    params = best_params or {
        "n_estimators": 150,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # Train on all verified training trips.
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(**params),
    )

    model.fit(X, y)

    metadata = {
        "training_rows": len(data),
        "training_trips": int(data["trip_id"].nunique()),
        "hyperparameters": params,
    }

    export_model(
        model,
        "RandomForest",
        metadata,
    )

    return model


if __name__ == "__main__":

    print("PathSnap model export module")

    print(
        "This module is ready. "
        "Run final training after model comparison and tuning."
    )
