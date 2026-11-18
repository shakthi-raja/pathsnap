
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = RESULTS_DIR / "figures"


def load_results():

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    folds_path = RESULTS_DIR / "cv_fold_results.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(
            "model_comparison.csv not found. "
            "Run model training first."
        )

    if not folds_path.exists():
        raise FileNotFoundError(
            "cv_fold_results.csv not found. "
            "Run model training first."
        )

    comparison = pd.read_csv(comparison_path)
    folds = pd.read_csv(folds_path)

    return comparison, folds


def plot_model_comparison(comparison):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    required = {"model", "macro_f1_mean", "log_loss_mean"}

    if not required.issubset(comparison.columns):
        raise ValueError(
            f"Comparison file must contain: {required}"
        )

    # Macro F1 comparison.
    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison["model"],
        comparison["macro_f1_mean"],
    )

    plt.title("Model Comparison - Macro F1")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Mean Macro F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_macro_f1.png",
        dpi=300,
    )

    plt.close()

    # Log Loss comparison.
    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison["model"],
        comparison["log_loss_mean"],
    )

    plt.title("Model Comparison - Log Loss")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Mean Log Loss")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_log_loss.png",
        dpi=300,
    )

    plt.close()


def plot_cross_validation(folds):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    required = {"model", "fold", "macro_f1"}

    if not required.issubset(folds.columns):
        raise ValueError(
            f"Fold results must contain: {required}"
        )

    plt.figure(figsize=(10, 6))

    for model_name, group in folds.groupby("model"):

        group = group.sort_values("fold")

        plt.plot(
            group["fold"],
            group["macro_f1"],
            marker="o",
            label=model_name,
        )

    plt.title("Cross-Validation Macro F1 Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "cross_validation_scores.png",
        dpi=300,
    )

    plt.close()


def generate_all_plots():

    print("Loading model evaluation results...")

    comparison, folds = load_results()

    print("Generating graphs...")

    plot_model_comparison(comparison)

    plot_cross_validation(folds)

    print("\nGraphs saved in:", FIGURES_DIR)


if __name__ == "__main__":

    generate_all_plots()
