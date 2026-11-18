from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from src.data.features import FEATURE_COLUMNS

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed/train")
OUT_DIR = Path("results/ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    files = [
        p for p in DATA_DIR.glob("*.parquet")
        if p.stem.isdigit() and len(p.stem) == 8
    ]
    frames = []
    for p in files:
        d = pd.read_parquet(p)
        d = d[d["training_eligible"] == True].copy()
        d["trip_id"] = p.stem
        frames.append(d)

    data = pd.concat(frames, ignore_index=True)
    data["label"] = data["label"].astype(int)
    return data

def main():
    print("Loading training data...", flush=True)
    data = load_data()

    X = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = data["label"]
    groups = data["trip_id"]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Training rows: {len(train_idx):,}", flush=True)
    print(f"Validation rows: {len(test_idx):,}", flush=True)
    print("Training models...", flush=True)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=20, class_weight="balanced", random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=250, max_depth=7, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", n_jobs=-1, random_state=42
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=200, max_leaf_nodes=31, random_state=42
            ))
        ])
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "seconds": round(time.time() - start, 2)
        })
        print(f"{name} completed.", flush=True)

    print("\nCNN requires the sequence-level training pass.", flush=True)
    print("Saving baseline results...", flush=True)
    pd.DataFrame(results).to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved to {OUT_DIR / 'model_comparison.csv'}")

if __name__ == "__main__":
    main()