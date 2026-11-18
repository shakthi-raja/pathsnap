
from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_candidates.parquet"

RESULTS_DIR = PROJECT_ROOT / "results"

VERIFIED_STATUS = "verified"

FEATURES = [
    "distance_m",
    "heading_diff_deg",
    "speed_mps",
    "dt_s",
    "candidate_rank",
]

RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_training_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}\n"
            "Wait for Person 1 to provide train_candidates.parquet."
        )

    print("Loading training dataset...")

    data = pd.read_parquet(DATA_PATH)

    required_columns = (
        FEATURES
        + ["trip_id", "label", "label_status"]
    )

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    print("\nDataset shape:", data.shape)

    print("\nAvailable label statuses:")
    print(data["label_status"].value_counts(dropna=False))

    # Keep only verified labels.
    data = data[
        data["label_status"] == VERIFIED_STATUS
    ].copy()

    if data.empty:
        raise ValueError(
            f"No rows found with label_status = "
            f"'{VERIFIED_STATUS}'. Confirm the label convention "
            "with Person 1."
        )

    # Validate labels.
    if data["label"].isnull().any():
        raise ValueError("Verified rows contain missing labels.")

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if data["trip_id"].isnull().any():
        raise ValueError("Some rows have missing trip IDs.")

    if data["trip_id"].nunique() < N_SPLITS:
        raise ValueError(
            "Not enough unique trips for 5-fold GroupKFold."
        )

    if data["label"].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes: 0 and 1."
        )

    print("\nVerified training rows:", len(data))
    print("Unique trips:", data["trip_id"].nunique())
    print("\nLabel distribution:")
    print(data["label"].value_counts())

    return data


# --------------------------------------------------
# 3. DEFINE THE FIVE ML MODELS
# --------------------------------------------------

def get_models():

    models = {

        "GaussianNB": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GaussianNB(),
        ),

        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),

        "DecisionTree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),
        ),

        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=150,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),

        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
        ),
    }

    return models


# --------------------------------------------------
# 4. TRAIN AND EVALUATE USING GROUPKFOLD
# --------------------------------------------------

def train_and_compare(data):

    X = data[FEATURES].copy()
    y = data["label"].astype(int)
    groups = data["trip_id"].astype(str)

    cv = GroupKFold(n_splits=N_SPLITS)

    models = get_models()

    fold_results = []

    print("\nStarting model comparison...\n")

    for model_name, model in models.items():

        print(f"Training {model_name}...")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups),
            start=1,
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Every fold must train on both classes.
            if y_train.nunique() != 2:
                raise ValueError(
                    f"{model_name}, fold {fold}: "
                    "training fold does not contain both classes."
                )

            # Train a fresh model for this fold.
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            probabilities = model.predict_proba(X_val)

            # Find probability column corresponding to class 1.
            classes = model.classes_.tolist()

            if 0 not in classes or 1 not in classes:
                raise ValueError(
                    f"{model_name}, fold {fold}: "
                    "model did not learn both classes."
                )

            positive_index = classes.index(1)

            positive_probabilities = probabilities[
                :, positive_index
            ]

            # Calculate evaluation metrics.
            macro_f1 = f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0,
            )

            loss = log_loss(
                y_val,
                probabilities,
                labels=model.classes_,
            )

            accuracy = (predictions == y_val).mean()

            fold_results.append({

                "model": model_name,
                "fold": fold,
                "macro_f1": macro_f1,
                "log_loss": loss,
                "accuracy": accuracy,
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "train_trips": groups.iloc[train_idx].nunique(),
                "validation_trips": groups.iloc[val_idx].nunique(),

            })

            print(
                f"  Fold {fold}: "
                f"Macro F1 = {macro_f1:.4f}, "
                f"Log Loss = {loss:.4f}"
            )

    return pd.DataFrame(fold_results)


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

def save_results(fold_results):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual fold results.
    fold_path = RESULTS_DIR / "cv_fold_results.csv"

    fold_results.to_csv(fold_path, index=False)

    # Calculate average performance for each model.
    comparison = (
        fold_results
        .groupby("model")
        .agg({
            "macro_f1": ["mean", "std"],
            "log_loss": ["mean", "std"],
            "accuracy": ["mean", "std"],
        })
    )

    comparison.columns = [
        "_".join(column)
        for column in comparison.columns
    ]

    comparison = comparison.reset_index()

    comparison = comparison.sort_values(
        "macro_f1_mean",
        ascending=False,
    )

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    comparison.to_csv(comparison_path, index=False)

    print("\nMODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))

    print("\nResults saved:")
    print(fold_path)
    print(comparison_path)


# --------------------------------------------------
# 6. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 60)
    print("PATHSNAP - ML MODEL TRAINING")
    print("=" * 60)

    data = load_training_data()

    results = train_and_compare(data)

    save_results(results)

    print("\nModel comparison completed successfully!")


if __name__ == "__main__":
    main()
