
import json

import optuna
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from src.ml.train_models import (
    FEATURES,
    RANDOM_STATE,
    N_SPLITS,
    RESULTS_DIR,
    load_training_data,
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

N_TRIALS = 25


# --------------------------------------------------
# 2. DEFINE OPTUNA OBJECTIVE
# --------------------------------------------------

def objective(trial, X, y, groups):

    # Optuna suggests hyperparameter values.
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 80, 300
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 4, 25
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 12
        ),

        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        ),
    }

    cv = GroupKFold(n_splits=N_SPLITS)

    fold_scores = []

    for train_idx, val_idx in cv.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if y_train.nunique() != 2:
            raise ValueError(
                "A training fold does not contain both classes."
            )

        model = make_pipeline(
            SimpleImputer(strategy="median"),

            RandomForestClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        score = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        fold_scores.append(score)

    return sum(fold_scores) / len(fold_scores)


# --------------------------------------------------
# 3. RUN BAYESIAN OPTIMISATION
# --------------------------------------------------

def run_optimization():

    print("=" * 60)
    print("PATHSNAP - OPTUNA HYPERPARAMETER OPTIMISATION")
    print("=" * 60)

    data = load_training_data()

    X = data[FEATURES].copy()

    y = data["label"].astype(int)

    groups = data["trip_id"].astype(str)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
    )

    study.optimize(
        lambda trial: objective(
            trial, X, y, groups
        ),
        n_trials=N_TRIALS,
    )

    # --------------------------------------------------
    # 4. SAVE OPTIMISATION RESULTS
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials_path = RESULTS_DIR / "optuna_trials.csv"

    study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    ).to_csv(
        trials_path,
        index=False,
    )

    best_results = {
        "model": "RandomForest",
        "best_macro_f1": study.best_value,
        "best_parameters": study.best_params,
        "number_of_trials": N_TRIALS,
        "random_state": RANDOM_STATE,
        "optimization_metric": "macro_f1",
    }

    params_path = RESULTS_DIR / "best_params.json"

    with open(params_path, "w") as file:
        json.dump(
            best_results,
            file,
            indent=4,
        )

    print("\nBEST RESULTS")
    print("=" * 60)

    print("Best Macro F1:", study.best_value)
    print("Best parameters:", study.best_params)

    print("\nSaved:")
    print(trials_path)
    print(params_path)

    return study


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

if __name__ == "__main__":
    run_optimization()
