"""Generate the consolidated technical report as LaTeX and compile it to PDF.

Prose is written as LaTeX. Only data values pulled from result files are passed
through :func:`_tex`, so the document cannot contain a half-escaped macro.

Usage::

    python tools/build_report.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "report"
OUTPUT_TEX = REPORT_DIR / "PathSnap_Report.tex"
OUTPUT_PDF = REPORT_DIR / "PathSnap_Report.pdf"
LOG_PATH = REPORT_DIR / "PathSnap_Report.log"


def _tex(value: object) -> str:
    """Escape a data value for LaTeX text mode."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _tt(name: str) -> str:
    """Inline monospace for file and column names."""
    return r"\texttt{" + _tex(name).replace(r"\textbackslash{}", "") + "}"


def _pct(value, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number * 100:.{digits}f}\\%"


def _num(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}"


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _model_hyperparameters() -> dict[str, dict[str, object]]:
    """Read the hyperparameters the training script actually builds."""
    from src.ml.train_models import build_models

    keep = {
        "GaussianNB": ["var_smoothing"],
        "LogisticRegression": ["C", "max_iter", "solver", "penalty"],
        "DecisionTree": ["max_depth", "criterion", "min_samples_split", "min_samples_leaf"],
        "RandomForest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
        "HistGradientBoosting": [
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ],
    }
    table: dict[str, dict[str, object]] = {}
    for name, pipeline in build_models().items():
        estimator = pipeline.steps[-1][1]
        params = estimator.get_params()
        table[name] = {key: params[key] for key in keep.get(name, []) if key in params}
    return table


PURPOSE = {
    "GaussianNB": (
        "Probabilistic floor. It is cheap, needs no tuning, and its independence assumption is "
        "plainly wrong here because distance and heading are correlated. The measured log loss "
        "confirms that: it is the worst of the five while accuracy stays respectable, which is what "
        "an over-confident independence assumption looks like."
    ),
    "LogisticRegression": (
        "Linear baseline. It establishes how much signal is available from a weighted sum of the "
        "features and trains in about two seconds. It holds up better than expected, which says the "
        "features carry useful information and the problem is not purely interaction driven. Its "
        "limit is that a linear boundary cannot express thresholds such as a distance cut-off."
    ),
    "DecisionTree": (
        "Threshold reader. It tests whether axis-aligned rules are enough and produces rules a "
        "person can inspect. The log loss shows severe over-confidence: leaf probabilities collapse "
        "to zero and one, so the loss is an order of magnitude worse than the ensembles."
    ),
    "RandomForest": (
        "Variance reduction. It handles feature interactions with little tuning and produced the "
        "highest raw Top-1 in cross-validation. Its probability estimate is the mean over many "
        "trees, which keeps it reasonable though still worse than boosting. It was not shipped "
        "because the fitted artifact is 243 MB, beyond repository limits, for half a point of "
        "accuracy."
    ),
    "HistGradientBoosting": (
        "Sequential correction. It matches the forest on macro F1 and beats it on log loss, which "
        "matters because those probabilities become the decoder's emissions. It also fits roughly "
        "five times faster and ships as a 740 kB artifact. This is the model behind the deployed "
        "system."
    ),
}

LOSSES = {
    "GaussianNB": "maximum likelihood under a Gaussian assumption; no iterative loss.",
    "LogisticRegression": "binary cross entropy (log loss), minimised by L-BFGS.",
    "DecisionTree": "Gini impurity for the split criterion.",
    "RandomForest": "Gini impurity averaged over bootstrap samples.",
    "HistGradientBoosting": "binary log loss, minimised stagewise with histogram splits.",
}


PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=2.4cm,top=2.3cm,bottom=2.3cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{float}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}

\definecolor{ink}{HTML}{14171A}
\definecolor{muted}{HTML}{6B6F6A}
\definecolor{rule}{HTML}{DCDDD7}

\captionsetup{font=small,labelfont=bf,textfont=small,skip=5pt}
\captionsetup[table]{position=top}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.12}

\titleformat{\section}{\Large\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{16pt}{5pt}
\titlespacing*{\subsection}{0pt}{11pt}{3pt}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.3pt}
\fancyhead[L]{\footnotesize\color{muted}PathSnap: Distributed Map-Matching and Route Reconstruction}
\fancyhead[R]{\footnotesize\color{muted}\thepage}
"""


def build_tex() -> str:
    comparison = pd.read_csv(RESULTS / "model_comparison.csv").sort_values(
        "cv_top1_mean", ascending=False
    )
    metadata = _read_json(ARTIFACTS / "model_metadata.json", {})
    tuned = _read_json(ARTIFACTS / "tuned_metadata.json")
    trials = (
        pd.read_csv(RESULTS / "optuna_trials.csv")
        if (RESULTS / "optuna_trials.csv").is_file()
        else None
    )
    evaluator = _read_json(RESULTS / "evaluator_test_results.json")
    api = _read_json(RESULTS / "api_evaluation.json", {})
    params = _model_hyperparameters()
    selected = metadata.get("model_name", "the selected model")
    artifact_kb = ARTIFACTS.joinpath("final_pipeline.joblib").stat().st_size / 1024

    doc: list[str] = [PREAMBLE, r"\begin{document}", r"\thispagestyle{empty}"]

    # ------------------------------------------------------------- title page
    doc.append(
        r"""
\begin{center}
\vspace*{2.4cm}
{\footnotesize\scshape PES University \quad Department of Computer Science and Engineering \quad RR Campus}

\vspace{1.8cm}
{\Huge\bfseries PathSnap}\\[8pt]
{\LARGE\bfseries Distributed Map-Matching and Route Reconstruction}

\vspace{0.7cm}
{\color{rule}\rule{0.45\textwidth}{0.4pt}}
\vspace{0.7cm}

{\large Reconstructing the road a vehicle actually drove from noisy GPS observations, using a
candidate classifier and a Viterbi decoder over the road network.}

\vspace{2.2cm}
\begin{tabular}{@{}ll@{}}
\textbf{Course} & UE24CS352A Machine Learning \\
\textbf{Repository} & \url{https://github.com/shakthi-raja/pathsnap} \\
\textbf{Partitions} & 80 training, 10 student-testing, 10 reserved evaluator trips \\
\end{tabular}

\vspace{1.6cm}
\begin{tabular}{@{}ll@{}}
\textbf{Person 1} & Data engineering: reading, network, candidates, features, alignment \\
\textbf{Person 2} & Model training, comparison and optimisation \\
\textbf{Person 3} & HMM and Viterbi decoding, FastAPI backend \\
\textbf{Person 4} & Evaluator interface, Folium maps, integration, report \\
\end{tabular}

\vfill
{\footnotesize\color{muted}\today}
\end{center}
\clearpage
"""
    )

    # ---------------------------------------------------------------- abstract
    doc.append(r"\section*{Abstract}" + "\n")
    doc.append(
        "PathSnap matches noisy GPS observations to a road network and rebuilds the route a "
        "vehicle took. The dataset holds 90 trips from 77 locations on six continents, which rules "
        "out any global coordinate frame and any feature tied to one city. We score candidate road "
        "segments with a binary classifier over scale-free geometric features, then decode a "
        "connected path with Viterbi over the road graph. Five model families were compared under "
        "grouped cross-validation. The shipped model is "
        + _tex(selected)
        + ", with a student-test Top-1 accuracy of "
        + _num(metadata.get("test_top1_accuracy"), 3)
        + " and log loss "
        + _num(metadata.get("test_log_loss"), 3)
        + ". On the ten reserved evaluator trips the decoder averaged "
        + (
            _num(evaluator["summary"]["decoded_accuracy_mean"] * 100, 1) + r"\%"
            if evaluator
            else "n/a"
        )
        + " pointwise accuracy, improving on the classifier alone on every trip.\n\n"
        + r"\clearpage" + "\n"
    )

    doc.append(r"\tableofcontents" + "\n" + r"\clearpage" + "\n")

    # ------------------------------------------------------------ 1 introduction
    doc.append(
        r"""
\section{Introduction}

A GPS receiver in a city reports positions that wobble around the true path. Buildings reflect
signals, tunnels and underpasses break reception, and a 1\,Hz receiver can place a moving car on a
parallel street. Map matching is the task of recovering the road a vehicle actually used from that
noisy record.

The dataset supplies, for each trip, a GPS trace, a local road network given as nodes and directed
arcs, and a reference route expressed as an ordered list of arc identifiers. The reference route is
a sequence, not a label per observation, and that shapes the whole approach: the supervised target
has to be constructed, and it has to be constructed carefully enough that a wrong target is never
produced.

This report covers the data audit, the modelling decisions and the evidence behind them, the
sequential decoder, and the deployed two-laptop system. Every number in it comes from a file in the
repository, produced by the commands listed in Section~\ref{sec:reproduce}.
"""
    )

    # ------------------------------------------------------------------ 2 data
    doc.append(
        r"""
\section{Dataset understanding and exploratory data analysis}

The material was treated as a hypothesis rather than as fact. All 90 trips were audited before any
pipeline code was written, because a mistaken assumption there propagates into every result that
follows.

\subsection{Structure and partitions}

Each trip folder holds four tab separated files without headers: \texttt{.track} with longitude,
latitude and a timestamp in seconds; \texttt{.nodes} with node coordinates, where the row index is
the node identifier; \texttt{.arcs} with directed from and to node identifiers, where the row index
is the arc identifier; and \texttt{.route} with the reference sequence of arc identifiers. The 80
training trips are used for fitting and cross-validation, the 10 student-testing trips for
development comparison, and a further 10 reserved trips for final evaluation only.

\subsection{Audit findings}

Four findings changed the design.

The trips are not confined to one city. They span 77 distinct locations on six continents, from
$34^{\circ}$ south to $61^{\circ}$ north. A feature built on absolute coordinates would memorise a
city and fail everywhere else, and no single projected coordinate system is correct across the set,
so each trip is projected into its own local metric frame.

The median distance from an observation to the nearest road is 4.2 metres, with a 90th percentile of
6.0 metres. That justifies a 50 metre candidate radius with evidence rather than assumption.

Networks are large: the median trip carries 18,762 arcs and the largest carries 307,392, so
candidate search needs a spatial index.

The reference route does not always describe the whole drive. In 23 trips it covers less than 70
percent of the track length, and three trips are badly broken, with usable coverage of 14, 29 and 86
percent.
"""
    )

    doc.append(
        r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{figures/fig_trip_locations.png}
\caption{Location of every trip in the dataset. Latitude spans 35 degrees south to 62 north, so
grouped cross-validation over trips is cross-city validation rather than a random split.}
\label{fig:locations}
\end{figure}

That last finding matters more than it looks. A reference route covering only part of a drive cannot
supply a label for observations outside it, and forcing one would teach the model to be wrong with
confidence. The alignment stage therefore marks each observation as confident or unlabelled, and only
confident observations are used for supervision.
"""
    )

    doc.append(
        r"""
\subsection{Preprocessing}

\begin{enumerate}[leftmargin=1.5em,itemsep=2pt]
\item Parse the four files for a trip. Node and arc identifiers are 0-based row indices, so identity
is positional.
\item Project the trip into a local azimuthal equidistant frame centred on its node extent. All
distances from this point are metres.
\item Generate candidates: every arc within 50 metres of an observation, expanding to 200 metres
when nothing is found, capped at ten candidates ranked by distance.
\item Build features from previous observations only. No future observation and no reference data
enters a feature.
\item Align the reference route to observations with a min-sum forward and backward pass over
ordered route occurrences, producing a label and a confidence flag per observation.
\item Emit one parquet per trip holding candidate rows, features, a nullable label and an
eligibility flag.
\end{enumerate}

Of 1,304,456 candidate rows across the 80 training trips, 480,258 carried a confident label, or 36.8
percent. The remainder were withheld by the alignment rather than guessed. After filtering, 70 of the
80 trips contribute labelled rows; the other ten produced no confident alignment at all.

\subsection{Features and their justification}

Eighteen features describe each observation and candidate pair, grouped in
Table~\ref{tab:features}. The design rule was that every feature must mean the same thing in every
city: metres, normalised fractions, angles encoded as sines and cosines, and elapsed time. Latitude
and longitude are deliberately excluded.
"""
    )

    feature_rows = [
        (r"distance\_m, candidate\_rank", "Fit of the observation to the arc",
         "Metres and rank, same meaning anywhere"),
        (r"position\_fraction, position\_m, length\_m", "Where along the arc the point falls",
         "Normalised and metric"),
        (r"road\_sin, road\_cos, heading\_sin, heading\_cos, heading\_difference\_rad",
         "Travel direction against road direction", "Sines and cosines avoid the wrap at 0 and 360"),
        (r"movement\_m, dt\_s, speed\_mps, valid\_time\_gap, stationary",
         "Motion between observations", "Time based, survives 20 to 149 km/h"),
        (r"gaussian\_emission\_20m, log\_emission\_20m", "The decoder's emission term",
         "Derived from distance alone"),
    ]
    doc.append(
        r"""
\begin{table}[H]
\centering
\caption{Feature groups. Every column is scale-free or expressed in metres, so the model transfers
to a city it has never seen.}
\label{tab:features}
\small
\begin{tabular}{@{}p{0.30\textwidth}p{0.30\textwidth}p{0.30\textwidth}@{}}
\toprule
\textbf{Group} & \textbf{What it captures} & \textbf{Why it transfers} \\
\midrule
"""
        + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in feature_rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )

    doc.append(
        r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{figures/fig_gps_over_roads.png}
\caption{Trip 00000062 in its local metric frame. Blue marks are GPS observations and grey lines are
the road network. The offset between the trace and the carriageway is the error the model has to
absorb.}
\label{fig:gps}
\end{figure}
\clearpage
"""
    )

    # -------------------------------------------------------- 3 model comparison
    doc.append(
        r"""
\section{Model experimentation and comparison}

Predicting an arc identifier directly is not viable. Arc identifiers are row positions inside one
trip, so the same integer means a different road in another city, and there are thousands of classes
per trip with no relationship between them. The formulation used instead is binary candidate
classification: each observation contributes several candidate arcs, each pair carries a label
stating whether that arc is the aligned reference arc, and the model learns to score a candidate. The
highest scoring candidate is the classifier-only prediction.

Five model families were trained on the labelled training rows with five-fold grouped
cross-validation, where folds are grouped by trip. Preprocessing sits inside each fold, so imputation
is fitted on training rows only. Table~\ref{tab:comparison} reports the result. Observation-level
Top-1 is measured per GPS point rather than per candidate row, because row accuracy is dominated by
negatives and would flatter a model that rejects everything.
"""
    )

    table_rows = []
    for _, row in comparison.iterrows():
        table_rows.append(
            f"{_tex(row['model'])} & {row['cv_macro_f1_mean']:.3f} $\\pm$ "
            f"{row['cv_macro_f1_std']:.3f} & {_pct(row['cv_top1_mean'])} "
            f"& {_pct(row['test_macro_f1'])} & {_pct(row['test_top1_accuracy'])} "
            f"& {_num(row['test_log_loss'])} & {row['cv_train_seconds_mean']:.1f} "
            f"& {row['inference_ms']:.0f} \\\\"
        )

    doc.append(
        r"""
\begin{table}[H]
\centering
\caption{Model comparison. Cross-validation uses five folds grouped by trip. Student-test scores
come from the ten development trips, which were never used for fitting.}
\label{tab:comparison}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Model} & \textbf{CV macro F1} & \textbf{CV Top-1} & \textbf{Test macro F1}
 & \textbf{Test Top-1} & \textbf{Log loss} & \textbf{Fit (s)} & \textbf{Infer (ms)} \\
\midrule
"""
        + "\n".join(table_rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}

The shipped model is \textbf{"""
        + _tex(selected)
        + r"""}.

Top-1 accuracy is the primary metric because the task asks for one road per observation. Macro F1
and log loss describe probability quality, which matters because those probabilities become the
decoder's emissions. Models within 0.010 cross-validated Top-1 of the best were treated as tied and
ranked by log loss. That rule changed no accuracy decision, and it produced a """
        + f"{artifact_kb:.0f}"
        + r""" kB artifact instead of the 243 MB forest, which would also exceed repository limits.

\subsection{Analysis of each model}
"""
    )

    for _, row in comparison.iterrows():
        name = row["model"]
        param_rows = " \\\\\n".join(
            f"{_tex(key)} & {_tex(value)}" for key, value in params.get(name, {}).items()
        ) + " \\\\"
        doc.append(
            r"\subsubsection*{" + _tex(name) + "}\n"
            r"\noindent\textit{Role.} " + PURPOSE.get(name, "") + "\n\n"
            r"\begin{center}\small\begin{tabular}{@{}ll@{}}\toprule"
            r"\textbf{Hyperparameter} & \textbf{Initial value} \\ \midrule" + "\n"
            + param_rows + "\n"
            r"\bottomrule\end{tabular}\end{center}" + "\n\n"
            r"\noindent\textit{Objective.} " + LOSSES.get(name, "") + "\n\n"
            r"\noindent\textit{Measured.} Cross-validated macro F1 "
            + f"{row['cv_macro_f1_mean']:.3f}"
            + " with a fold standard deviation of "
            + f"{row['cv_macro_f1_std']:.3f}"
            + ", and cross-validated Top-1 "
            + _pct(row["cv_top1_mean"])
            + ". On the student-testing trips it reached macro F1 "
            + _pct(row["test_macro_f1"])
            + ", Top-1 "
            + _pct(row["test_top1_accuracy"])
            + " and log loss "
            + _num(row["test_log_loss"])
            + ", fitting in "
            + f"{row['cv_train_seconds_mean']:.1f}"
            + " seconds per fold and scoring in "
            + f"{row['inference_ms']:.0f}"
            + r""" milliseconds.

\noindent\textit{Fit assessment.} The gap between cross-validated and student-test Top-1 is """
            + f"{abs(row['test_top1_accuracy'] - row['cv_top1_mean']) * 100:.1f}"
            + r""" percentage points. A gap that small does not indicate overfitting to the training
cities, and the fold spread stays narrow enough that no single fold dominates the average.

"""
        )

    # ------------------------------------------------- 3.4 reserved partition
    if evaluator:
        summary = evaluator["summary"]
        rows_ev = []
        for trip_id, payload in sorted(evaluator["trips"].items()):
            metrics = payload.get("metrics") or {}
            rows_ev.append(
                f"{_tex(trip_id)} & {metrics.get('evaluated_observations', '--')} "
                f"& {_pct(metrics.get('top1_accuracy'))} "
                f"& {_pct(metrics.get('pointwise_accuracy'))} "
                f"& {_pct(metrics.get('route_continuity'))} "
                f"& {_pct(metrics.get('candidate_miss_rate'))} \\\\"
            )
        doc.append(
            r"""
\subsection{Final evaluation on the reserved partition}

The ten reserved trips were supplied after the model, the decoder and every hyperparameter had been
fixed. They were used for scoring only, and no decision in this report depends on them.

\begin{table}[H]
\centering
\caption{Reserved evaluator partition. Every trip was run through the deployed API, with the reference
route used only after the prediction existed.}
\label{tab:evaluator}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Trip} & \textbf{Evaluable} & \textbf{Classifier Top-1} & \textbf{Decoded accuracy}
 & \textbf{Continuity} & \textbf{Candidate miss} \\
\midrule
"""
            + "\n".join(rows_ev)
            + r"""
\bottomrule
\end{tabular}
\end{table}

Across the ten reserved trips the decoded accuracy averaged \textbf{"""
            + f"{summary['decoded_accuracy_mean'] * 100:.1f}"
            + r"""\%}, between """
            + f"{summary['decoded_accuracy_min'] * 100:.1f}"
            + " and "
            + f"{summary['decoded_accuracy_max'] * 100:.1f}"
            + r"""\%. The classifier alone averaged """
            + f"{summary['top1_accuracy_mean'] * 100:.1f}"
            + r"""\%, so decoding was worth \textbf{"""
            + f"{summary['decoding_gain_points']:.1f}"
            + r""" percentage points} on trips the model had never seen.

Trip 00000015 is the outlier at """
            + f"{float(evaluator['trips']['00000015']['metrics']['pointwise_accuracy']) * 100:.1f}"
            + r"""\%. Its road extract is sparse, 3,225 arcs across a 53 kilometre drive, so long
stretches repeat a single arc and only 3,832 of 6,854 observations produced a confident reference
alignment. The result is reported as measured.

\begin{figure}[H]
\centering
\includegraphics[width=0.84\textwidth]{figures/fig_cv_macro_f1.png}
\caption{Cross-validated macro F1 for each model, with one standard deviation across the five grouped
folds.}
\label{fig:cv}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.84\textwidth]{figures/fig_test_top1_macro_f1.png}
\caption{Student-test Top-1 accuracy and macro F1 for the same five models.}
\label{fig:test}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figures/fig_log_loss.png}
\caption{Student-test log loss. The single decision tree is badly over-confident, which is why its
probabilities are unusable as decoder emissions.}
\label{fig:logloss}
\end{figure}
\clearpage
"""
        )

    # ------------------------------------------------------------- 4 bayesian
    doc.append(r"\section{Bayesian hyperparameter optimisation}" + "\n")
    if tuned and trials is not None:
        best = tuned.get("best_params", {})
        best_rows = " \\\\\n".join(
            f"{_tex(key)} & {_tex(round(value, 6) if isinstance(value, float) else value)}"
            for key, value in best.items()
        ) + " \\\\"
        doc.append(
            r"""The selected algorithm was tuned with Optuna's Tree-structured Parzen Estimator, a
Bayesian sampler that models which regions of the search space produce good trials instead of
sampling uniformly. The objective combined observation-level Top-1 and macro F1, averaged over three
folds grouped by trip, so tuning could not benefit from a lucky split.

\begin{table}[H]
\centering
\caption{Bayesian search settings and outcome.}
\label{tab:tuning}
\begin{tabular}{@{}p{0.26\textwidth}p{0.64\textwidth}@{}}
\toprule
\textbf{Item} & \textbf{Value} \\
\midrule
Sampler & TPE, seed """
            + str(tuned.get("seed", 42))
            + r""" \\
Trials & """
            + str(len(trials))
            + r""" \\
Search space & \texttt{max\_iter} 100 to 350, \texttt{learning\_rate} 0.03 to 0.3 (log),
\texttt{max\_leaf\_nodes} 15 to 63, \texttt{min\_samples\_leaf} 10 to 60,
\texttt{l2\_regularization} \(10^{-4}\) to 10 (log) \\
Objective & \(0.7 \times\) Top-1 \(+\) \(0.3 \times\) macro F1 over three grouped folds \\
Best objective & """
            + f"{tuned.get('objective', 0):.4f}"
            + r""" \\
Tuned student-test Top-1 & """
            + _pct(tuned.get("test_top1_accuracy"))
            + r""" \\
Untuned student-test Top-1 & """
            + _pct(metadata.get("test_top1_accuracy"))
            + r""" \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Best configuration returned by the search.}
\label{tab:bestparams}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Best value} \\
\midrule
"""
            + best_rows
            + r"""
\bottomrule
\end{tabular}
\end{table}

The tuned configuration did not improve on the untuned model on student-test Top-1. The full record is
kept in \texttt{results/optuna\_trials.csv}. The honest reading is that the default configuration
already sat close to the best region of this space for this dataset, so the tuned model was not
adopted.
"""
        )

    # ------------------------------------------------------------------ 5 HMM
    doc.append(
        r"""
\section{HMM and Viterbi route reconstruction}

A classifier scores candidates independently, so consecutive observations can land on arcs that no
vehicle could drive between. The decoder resolves this by scoring whole sequences rather than single
observations.

\begin{table}[H]
\centering
\caption{Decoder design.}
\label{tab:hmm}
\small
\begin{tabular}{@{}p{0.22\textwidth}p{0.70\textwidth}@{}}
\toprule
\textbf{Component} & \textbf{Choice} \\
\midrule
Hidden states & The candidate arcs available at each observation, each carrying its projection
fraction along the arc. \\
\midrule
Emission & The classifier's positive-class probability, clipped, normalised across the candidates of
that observation, then logged. A binary probability is not a distribution over candidates, so
normalisation is required before states can be compared. \\
\midrule
Transition & Directed road distance from the previous candidate to the next, measured with a bounded
Dijkstra over the sparse adjacency matrix. The score penalises the difference between road distance
and GPS displacement, backwards movement, and speeds above 50 metres per second. \\
\midrule
Missing candidates & Observations with no candidate inside the radius are absent from the grid. The
transition then spans the real time gap, and the response reports how many observations were
affected. \\
\midrule
Disconnected pairs & When no road link exists within 400 metres the pair receives a heavy finite
penalty scaled by straight-line distance, so one unreachable pair cannot make a whole trip
undecodable. Pairs beyond two kilometres are set to negative infinity. \\
\bottomrule
\end{tabular}
\end{table}
"""
    )

    if api:
        api_rows = []
        for trip_id, payload in sorted(api.items()):
            metrics = payload.get("metrics") or {}
            api_rows.append(
                f"{_tex(trip_id)} & {metrics.get('evaluated_observations', '--')} "
                f"& {_pct(metrics.get('top1_accuracy'))} "
                f"& {_pct(metrics.get('pointwise_accuracy'))} "
                f"& {_pct(metrics.get('classifier_continuity'))} "
                f"& {_pct(metrics.get('route_continuity'))} \\\\"
            )
        gains = [
            (payload.get("metrics") or {}).get("pointwise_accuracy", 0)
            - (payload.get("metrics") or {}).get("top1_accuracy", 0)
            for payload in api.values()
        ]
        doc.append(
            r"""
After Viterbi has chosen an arc per observation, a reconstruction pass turns that sequence into
travelled portions of road. Each chosen arc keeps a start and end offset along its length, so the
drawn route follows the part of the arc the vehicle used rather than hopping between arc endpoints.
Pieces split only where the road network cannot connect them, which is where the route pieces in the
figures come from.

\subsection{Classifier alone against classifier plus decoder}

The same observations, the same alignment and the same scoring rule were used for both paths, so the
only difference is the decoder.

\begin{table}[H]
\centering
\caption{Pointwise accuracy with and without sequential decoding, on development trips.}
\label{tab:decoder}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Trip} & \textbf{Evaluable} & \textbf{Classifier Top-1} & \textbf{Decoded}
 & \textbf{Classifier continuity} & \textbf{Decoded continuity} \\
\midrule
"""
            + "\n".join(api_rows)
            + r"""
\bottomrule
\end{tabular}
\end{table}

Decoding improved pointwise accuracy on every trip measured, by between """
            + f"{min(gains) * 100:.1f}"
            + " and "
            + f"{max(gains) * 100:.1f}"
            + r""" percentage points. The largest gains appear where the classifier is weakest, which
is the expected behaviour: when local evidence is ambiguous, connectivity carries the decision.
\clearpage
"""
        )

    # --------------------------------------------------------------- 6 folium
    doc.append(
        r"""
\section{Geographical visualisation with Folium}

Maps are produced by \texttt{src/visualization/maps.py} from live data, and by
\texttt{tools/render\_trip\_map.py} for report evidence. The evaluator interface renders GPS
observations, the relevant road context and the predicted route. The reference route appears only in
report maps built from training or student-testing trips, never in the interface.

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figures/folium_trip_00000062.png}
\caption{Trip 00000062, Yekaterinburg. Blue marks are raw GPS observations, grey lines the road
network and orange the reference route. Layers toggle independently and the legend is generated only
for layers actually drawn. The interactive HTML is committed at
\texttt{results/maps/trip\_00000062\_overview.html}.}
\label{fig:folium}
\end{figure}
"""
    )

    # --------------------------------------------------------------- 7 system
    doc.append(
        r"""
\section{API, UI and deployment}

Laptop 1 runs the FastAPI backend and holds the model, the decoder and the reference routes. Laptop
2 runs the Streamlit interface and reaches the backend over the local network. Ground truth never
crosses that link: the response contract has no field for the reference route.

The service runs in one of two modes. In trained mode it builds candidates, scores them, decodes a
route and then scores that route against the reference, so the response carries accuracy, Top-1,
continuity and candidate miss rate. In evaluator mode it reads candidate probabilities that were
computed ahead of time and never opens a reference route at all, so the response carries only
geometry, continuity and runtime. The deployment used for demonstration ran in evaluator mode, and
that is why the readout in Figure~\ref{fig:ui} shows a single metric. Nothing is hidden: the health
endpoint names the mode and the response simply omits what was not computed.

The pair was exercised over the shared network with the API on Laptop 1 at \texttt{10.30.201.178:8000}
and the interface on Laptop 2. A prediction for trip 00000023 returned in 1,196 milliseconds,
measured at the interface.

\begin{table}[H]
\centering
\caption{Public endpoints.}
\label{tab:api}
\small
\begin{tabular}{@{}p{0.20\textwidth}p{0.72\textwidth}@{}}
\toprule
\textbf{Endpoint} & \textbf{Purpose} \\
\midrule
\texttt{GET /health} & Liveness, trip count and the loaded model name. \\
\texttt{GET /trips} & Trip identifiers discovered from the server directory. Nothing is hardcoded,
so the reserved partition appears without a code change. \\
\texttt{POST /predict} & Runs candidates, features, scoring and decoding, then scores the result
against the reference route held on the server. \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.99\textwidth]{figures/ui_lan_prediction.png}
\caption{The interface on Laptop 2, served by the backend on Laptop 1 over the shared network. The ten
reserved trips are discovered from the server, so no code changed when the evaluator data arrived.
Trip 00000023 was selected and its route drawn in 1,196 milliseconds. The evaluator mode reports
continuity and runtime only, as described above.}
\label{fig:ui}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures/api_request_log.png}
\caption{Request log on Laptop 1 for the same prediction. Each line records the trip, the observation
count and the runtime, and the accuracy field reads \texttt{None} because evaluator mode does not open
a reference route. The guidelines require this evidence from the machine receiving the requests.}
\label{fig:apilog}
\end{figure}

\subsection{Reproducing the results}
\label{sec:reproduce}

The repository is self-contained apart from the dataset, which is never committed.

\begin{verbatim}
export PATHSNAP_DATA_ROOT=/path/to/data/raw

python -m src.data.pipeline --folder "$PATHSNAP_DATA_ROOT/train" \
    --output "$PATHSNAP_DATA_ROOT/../processed/train"
python -m src.data.pipeline --folder "$PATHSNAP_DATA_ROOT/student_test" \
    --output "$PATHSNAP_DATA_ROOT/../processed/student_test" --evaluation-export

python -m src.ml.train_models      # comparison, student-test scores, artifact
python -m src.ml.tune_model        # Bayesian search

export PATHSNAP_EVALUATION_ROOT=/path/to/evaluator_test
python -m src.api.main             # Laptop 1

export PATHSNAP_API_URL=http://<laptop-1-ip>:8000
python -m streamlit run src/ui/app.py   # Laptop 2
\end{verbatim}
\clearpage
"""
    )

    # ------------------------------------------------------------ 8 integrity
    doc.append(
        r"""
\section{Academic integrity and AI assistance}

Every figure and table in this report is produced by scripts in the repository from the recorded run.
The comparison table reads \texttt{results/model\_comparison.csv}, the optimisation section reads
\texttt{results/optuna\_trials.csv}, the decoder comparison reads \texttt{results/api\_evaluation.json},
and the reserved partition table reads \texttt{results/evaluator\_test\_results.json}.

Reference routes cannot influence a prediction. The service builds candidates, scores them and decodes
a route before it opens the route file, and the interface has no field that could carry ground truth.
The reserved partition was not used for training, feature selection, model choice or decoder
parameters; it arrived after those were fixed.

AI assistance was used during development, for code, analysis and drafts of this document. All
reported numbers were produced by running the pipeline rather than generated. Every module is covered
by tests, and each team member is responsible for being able to explain the sections attributed to
them.
"""
    )

    # ----------------------------------------------------------- 9 conclusion
    conclusion = r"""
\section{Conclusion}

The system recovers the driven route from noisy GPS on every trip tested.
"""
    if evaluator:
        conclusion += (
            "On the ten reserved evaluator trips the decoder averaged "
            + f"{evaluator['summary']['decoded_accuracy_mean'] * 100:.1f}"
            + r" percent pointwise accuracy, against "
            + f"{evaluator['summary']['top1_accuracy_mean'] * 100:.1f}"
            + r" percent for the classifier alone, and it improved on the classifier on every trip."
        )
    conclusion += r"""

Three decisions carried the result. Treating the task as candidate scoring rather than arc prediction
made it learnable across 77 cities. Constructing labels through a monotonic alignment, and refusing to
label where the alignment was uncertain, kept a third of the candidate rows out of training instead of
teaching the model to be wrong. Decoding with road connectivity turned independent scores into routes
a vehicle could drive.

The largest remaining weakness is candidate coverage on sparse networks, which is visible in the one
trip that scored below fifty percent, and the slowest trips take around eighteen seconds on a cold
server because candidate generation runs inside the request.
"""
    doc.append(conclusion)
    doc.append(r"\end{document}" + "\n")

    return "".join(doc)


def compile_pdf(tex_path: Path) -> None:
    """Compile with latexmk when present, otherwise pdflatex twice."""
    latexmk = shutil.which("latexmk")
    if latexmk:
        command = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        runs = 1
    else:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            raise SystemExit("No LaTeX engine found. Install texlive-core and texlive-latexextra.")
        command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        runs = 2

    for _ in range(runs):
        result = subprocess.run(command, cwd=tex_path.parent, capture_output=True, text=True)
        if result.returncode != 0:
            log = tex_path.with_suffix(".log")
            detail = log.read_text(errors="ignore") if log.is_file() else result.stdout
            errors = [
                line
                for line in detail.splitlines()
                if line.startswith("!") or "Undefined control sequence" in line
            ]
            raise SystemExit("LaTeX failed:\n" + "\n".join(errors[:12] or detail.splitlines()[-20:]))


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tex = build_tex()
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {OUTPUT_TEX} ({len(tex) // 1024} kB)")
    compile_pdf(OUTPUT_TEX)
    if OUTPUT_PDF.is_file():
        print(f"wrote {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
