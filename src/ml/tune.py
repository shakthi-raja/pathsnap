
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.data.features import FEATURE_COLUMNS

DATA_DIR = Path("data/processed/train")
RESULTS_DIR = Path("results/ml")
N_TRIALS = 10
RANDOM_STATE = 42
N_SPLITS = 3


def load_data():
    frames = []

    for path in DATA_DIR.glob("*.parquet"):
        if path.stem.isdigit() and len(path.stem) == 8:
            df = pd.read_parquet(path)
            df = df[df["training_eligible"] == True].copy()
            df = df[df["label"].notna()].copy()
            df["trip_id"] = path.stem
            frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    X = data[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    y = data["label"].astype(int)
    groups = data["trip_id"]

    return X, y, groups


def objective(trial, X, y, groups):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 250),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.2, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.1, 10.0, log=True
        ),
    }

    cv = GroupKFold(n_splits=N_SPLITS)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups)
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        score = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        scores.append(score)

        print(
            f"Trial {trial.number + 1}, "
            f"fold {fold + 1}: Macro F1 = {score:.4f}",
            flush=True,
        )

    return float(np.mean(scores))


def main():
    print("Loading training data...", flush=True)
    X, y, groups = load_data()

    print(f"Training rows: {len(X):,}", flush=True)
    print(f"Training trips: {groups.nunique()}", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
    )

    print("\nStarting XGBoost Bayesian optimisation...", flush=True)

    study.optimize(
        lambda trial: objective(trial, X, y, groups),
        n_trials=N_TRIALS,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study.trials_dataframe().to_csv(
        RESULTS_DIR / "optuna_trials.csv",
        index=False,
    )

    results = {
        "model": "XGBoost",
        "best_macro_f1": study.best_value,
        "best_parameters": study.best_params,
        "number_of_trials": N_TRIALS,
        "optimization_metric": "macro_f1",
        "random_state": RANDOM_STATE,
    }

    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nOPTIMISATION COMPLETE")
    print("Best Macro F1:", study.best_value)
    print("Best parameters:", study.best_params)
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
