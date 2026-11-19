"""Generate the report figures from recorded experiment outputs.

Every figure is drawn from a CSV or artifact produced by the training run. No
value in this script is typed in by hand, which is the property the report
needs: a reader can regenerate every chart from the repository.

Usage::

    python tools/make_report_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

INK = "#14171A"
MUTED = "#6B6F6A"
ROUTE = "#2F6B4F"
RULE = "#DCDDD7"
ACCENT = "#A63D2A"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _style(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def figure_cv(comparison: pd.DataFrame) -> None:
    """Cross-validated macro F1 per model with fold spread."""
    ordered = comparison.sort_values("cv_macro_f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        ordered["cv_macro_f1_mean"],
        xerr=ordered["cv_macro_f1_std"],
        color=ROUTE,
        height=0.55,
        error_kw={"ecolor": INK, "capsize": 3, "linewidth": 0.8},
    )
    ax.set_yticks(positions, ordered["model"])
    ax.set_xlabel("Macro F1 (mean of 5 grouped folds, error bar = 1 sd)")
    ax.set_xlim(0, 1)
    ax.set_title("Cross-validated macro F1 by model")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cv_macro_f1.png", dpi=200)
    plt.close(fig)


def figure_student_test(comparison: pd.DataFrame) -> None:
    """Student-test Top-1 and macro F1 side by side."""
    ordered = comparison.sort_values("test_top1_accuracy", ascending=True)
    positions = np.arange(len(ordered))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.barh(
        positions - width / 2,
        ordered["test_top1_accuracy"],
        height=width,
        color=ROUTE,
        label="Top-1 accuracy",
    )
    ax.barh(
        positions + width / 2,
        ordered["test_macro_f1"],
        height=width,
        color=MUTED,
        label="Macro F1",
    )
    ax.set_yticks(positions, ordered["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score on the ten student-testing trips")
    ax.set_title("Student-test performance")
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_test_top1_macro_f1.png", dpi=200)
    plt.close(fig)


def figure_log_loss(comparison: pd.DataFrame) -> None:
    """Log loss per model. Lower is better and it is what the decoder needs."""
    ordered = comparison.sort_values("test_log_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colours = [ACCENT if value > 0.6 else ROUTE for value in ordered["test_log_loss"]]
    ax.barh(ordered["model"], ordered["test_log_loss"], color=colours, height=0.55)
    ax.set_xlabel("Student-test log loss (lower is better)")
    ax.set_title("Probability quality")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_log_loss.png", dpi=200)
    plt.close(fig)


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")

    figure_cv(comparison)
    figure_student_test(comparison)
    figure_log_loss(comparison)
    print("wrote comparison figures")
    main_extras()
    return 0




# --------------------------------------------------------------------------
# figures added for the final report: quality, tuning, confusion, sequence
# --------------------------------------------------------------------------

def figure_data_quality() -> None:
    """Sampling intervals across the training trips, and how much got labelled."""
    quality = json.loads((RESULTS_DIR / "data_quality.json").read_text(encoding="utf-8"))
    gaps = quality["gaps"]

    buckets = {
        "1 s": gaps["gap_1s"],
        "2 to 5 s": gaps["gap_2_to_5s"],
        "6 to 60 s": gaps["gap_6_to_60s"],
        "over 60 s": gaps["gap_over_60s"],
    }
    confident = int(quality["labelled_observations"])
    uncertain = int(quality["withheld_observations"])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    names = list(buckets)
    values = [buckets[name]["count"] / 1e6 for name in names]
    axes[0].bar(names, values, color=[ROUTE, MUTED, MUTED, ACCENT], width=0.55)
    axes[0].set_ylabel("consecutive gaps (millions)")
    axes[0].set_title("Sampling intervals across 80 training trips")
    for index, name in enumerate(names):
        axes[0].text(
            index,
            values[index],
            f"{buckets[name]['share'] * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _style(axes[0])

    observations = confident + uncertain
    axes[1].bar(
        ["confident", "withheld"],
        [confident / 1e6, uncertain / 1e6],
        color=[ROUTE, ACCENT],
        width=0.5,
    )
    axes[1].set_ylabel("GPS observations (millions)")
    axes[1].set_title("Observations with a usable label")
    for index, value in enumerate([confident, uncertain]):
        axes[1].text(index, value / 1e6, f"{value / observations * 100:.1f}%",
                     ha="center", va="bottom", fontsize=8)
    _style(axes[1])

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_data_quality.png", dpi=200)
    plt.close(fig)


def figure_tuning_history() -> None:
    """Objective value per Optuna trial, with the running best."""
    trials = pd.read_csv(RESULTS_DIR / "optuna_trials.csv")
    trials = trials.dropna(subset=["value"]).sort_values("number")

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.scatter(trials["number"], trials["value"], s=28, color=MUTED, label="trial objective", zorder=3)
    running = trials["value"].cummax()
    ax.step(trials["number"], running, where="post", color=ROUTE, linewidth=2, label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("objective (0.7 x Top-1 + 0.3 x macro F1)")
    ax.set_title("Bayesian search history")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_optuna_history.png", dpi=200)
    plt.close(fig)


def figure_confusion() -> None:
    """Candidate-level confusion matrix for the shipped model."""
    import json as _json

    counts = _json.loads((RESULTS_DIR / "confusion_matrix.json").read_text(encoding="utf-8"))
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=matrix.max())
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            share = value / matrix.sum() * 100
            ax.text(
                column,
                row,
                f"{value:,}\n{share:.2f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if matrix[row, column] > matrix.max() * 0.5 else INK,
            )
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("predicted (candidate is the match)")
    ax.set_ylabel("actual")
    ax.set_title("Candidate classification, student test", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_confusion_matrix.png", dpi=200)
    plt.close(fig)


def figure_sequence() -> None:
    """Route-level transition F1, classifier only against decoded."""
    data = pd.read_csv(RESULTS_DIR / "sequence_metrics.csv", dtype={"trip_id": str})
    pivot = data.pivot_table(index="trip_id", columns="path", values="transition_f1")

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    positions = np.arange(len(pivot))
    width = 0.38
    ax.bar(positions - width / 2, pivot["classifier"], width, color=MUTED, label="classifier only")
    ax.bar(positions + width / 2, pivot["hmm"], width, color=ROUTE, label="with Viterbi decoding")
    ax.set_xticks(positions, pivot.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("transition F1")
    ax.set_ylim(0, 1)
    ax.set_title("Route-level agreement with the reference sequence")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_sequence_f1.png", dpi=200)
    plt.close(fig)


def figure_evaluator_trips() -> None:
    """Per-trip accuracy on the reserved partition."""
    import json as _json

    payload = _json.loads((RESULTS_DIR / "evaluator_test_results.json").read_text(encoding="utf-8"))
    trips = payload["trips"]
    ids = sorted(trips)
    classifier = [trips[t]["metrics"].get("top1_accuracy", 0) for t in ids]
    decoded = [trips[t]["metrics"].get("pointwise_accuracy", 0) for t in ids]

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    positions = np.arange(len(ids))
    width = 0.38
    ax.bar(positions - width / 2, classifier, width, color=MUTED, label="classifier only")
    ax.bar(positions + width / 2, decoded, width, color=ROUTE, label="after decoding")
    ax.set_xticks(positions, ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("pointwise accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Reserved evaluator partition, before and after decoding")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_evaluator_trips.png", dpi=200)
    plt.close(fig)


def main_extras() -> None:
    figure_data_quality()
    figure_tuning_history()
    figure_confusion()
    figure_sequence()
    figure_evaluator_trips()
    print("wrote quality, tuning, confusion and sequence figures")


if __name__ == "__main__":
    raise SystemExit(main())
