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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
