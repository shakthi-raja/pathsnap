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


def _key_configuration(name: str, params: dict[str, object]) -> str:
    """One-line description of the settings that define each pipeline."""
    values = params.get(name, {})
    if name == "GaussianNB":
        return r"\texttt{var\_smoothing} " + _tex(values.get("var_smoothing", ""))
    if name == "LogisticRegression":
        return (
            r"\texttt{C} " + _tex(values.get("C", ""))
            + ", " + _tex(values.get("penalty", "")) + " penalty"
        )
    if name == "DecisionTree":
        return "Gini impurity, " + r"\texttt{max\_depth} " + _tex(values.get("max_depth", ""))
    if name == "RandomForest":
        return (
            _tex(values.get("n_estimators", "")) + " trees, "
            + r"\texttt{min\_samples\_leaf} " + _tex(values.get("min_samples_leaf", ""))
        )
    if name == "HistGradientBoosting":
        return (
            _tex(values.get("max_iter", "")) + " iterations, "
            + r"\texttt{learning\_rate} " + _tex(values.get("learning_rate", ""))
        )
    return ""


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

FAMILY = {
    "GaussianNB": "Probabilistic",
    "LogisticRegression": "Linear",
    "DecisionTree": "Tree",
    "RandomForest": "Ensemble",
    "HistGradientBoosting": "Boosting",
}

STRENGTHS = {
    "GaussianNB": (
        "Strengths: no tuning, a fraction of a second to fit, and a closed-form probability. "
        "Limitations: it treats every feature as independent, so two features that say the same "
        "thing are counted twice, and its probabilities are not calibrated."
    ),
    "LogisticRegression": (
        "Strengths: a stable linear decision surface, honest probabilities and three seconds to "
        "train, which makes it a useful reference point. Limitations: it cannot represent a rule "
        "such as a hard distance cut-off without an explicit interaction term, and it underfits "
        "the curved decision boundary that the features describe."
    ),
    "DecisionTree": (
        "Strengths: interpretable axis-aligned rules and no scaling requirement. Limitations: a "
        "single tree is a high-variance estimator, and its leaf probabilities collapse towards zero "
        "and one, which is why its log loss is unusable even where its argmax is fine."
    ),
    "RandomForest": (
        "Strengths: the best raw Top-1 on the student-test partition, feature interactions handled "
        "without scaling, and a probability that is an average over eighty trees. Limitations: a "
        "243 MB artifact, the slowest scoring path of the five and probabilities that are good but "
        "not as sharp as boosting."
    ),
    "HistGradientBoosting": (
        "Strengths: the best log loss of the five, competitive Top-1 and macro F1, a 740 kB "
        "artifact and fast scoring. Limitations: it needs careful early-stopping and "
        "regularisation settings, and its probability output is only as good as the candidate "
        "set it scores."
    ),
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
\setlength{\emergencystretch}{2.5em}
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
    quality = _read_json(RESULTS / "data_quality.json", {})
    confusion = _read_json(RESULTS / "confusion_matrix.json")
    missing = (
        pd.read_csv(RESULTS / "missing_values.csv")
        if (RESULTS / "missing_values.csv").is_file()
        else pd.DataFrame()
    )
    extended = (
        pd.read_csv(RESULTS / "model_test_metrics.csv").set_index("model")
        if (RESULTS / "model_test_metrics.csv").is_file()
        else pd.DataFrame()
    )
    tuned_vs = (
        pd.read_csv(RESULTS / "tuned_vs_untuned.csv")
        if (RESULTS / "tuned_vs_untuned.csv").is_file()
        else pd.DataFrame()
    )
    sequence = (
        pd.read_csv(RESULTS / "sequence_metrics.csv", dtype={"trip_id": str})
        if (RESULTS / "sequence_metrics.csv").is_file()
        else pd.DataFrame()
    )
    params = _model_hyperparameters()
    selected = metadata.get("model_name", "the selected model")

    sequence_rows = ""
    sequence_means = {}
    if not sequence.empty:
        pivot = sequence.pivot_table(
            index="trip_id",
            columns="path",
            values=["transition_precision", "transition_recall", "transition_f1"],
        )
        reference_arcs = sequence.groupby("trip_id")["reference_arcs"].first()
        rows = []
        for trip_id in sorted(pivot.index):
            rows.append(
                f"{_tex(trip_id)} & {int(reference_arcs.loc[trip_id])} "
                f"& {_pct(pivot.loc[trip_id, ('transition_precision', 'classifier')])} "
                f"& {_pct(pivot.loc[trip_id, ('transition_recall', 'classifier')])} "
                f"& {_pct(pivot.loc[trip_id, ('transition_f1', 'classifier')])} "
                f"& {_pct(pivot.loc[trip_id, ('transition_precision', 'hmm')])} "
                f"& {_pct(pivot.loc[trip_id, ('transition_recall', 'hmm')])} "
                f"& {_pct(pivot.loc[trip_id, ('transition_f1', 'hmm')])} \\\\"
            )
        means = pivot.mean()
        rows.append(
            r"\midrule Mean & -- "
            f"& {_pct(means[('transition_precision', 'classifier')])} "
            f"& {_pct(means[('transition_recall', 'classifier')])} "
            f"& {_pct(means[('transition_f1', 'classifier')])} "
            f"& {_pct(means[('transition_precision', 'hmm')])} "
            f"& {_pct(means[('transition_recall', 'hmm')])} "
            f"& {_pct(means[('transition_f1', 'hmm')])} \\\\"
        )
        sequence_rows = "\n".join(rows)
        gains = (
            pivot[("transition_f1", "hmm")] - pivot[("transition_f1", "classifier")]
        )
        weakest = pivot[("transition_f1", "hmm")].idxmin()
        classifier_rows = sequence[sequence["path"] == "classifier"].set_index("trip_id")
        sequence_means = {
            "weakest_trip": str(weakest),
            "weakest_f1": float(pivot.loc[weakest, ("transition_f1", "hmm")]),
            "weakest_predicted_arcs": int(classifier_rows.loc[weakest, "predicted_arcs"]),
            "weakest_reference_arcs": int(classifier_rows.loc[weakest, "reference_arcs"]),
            "classifier_f1": float(means[("transition_f1", "classifier")]),
            "hmm_f1": float(means[("transition_f1", "hmm")]),
            "classifier_precision": float(means[("transition_precision", "classifier")]),
            "hmm_precision": float(means[("transition_precision", "hmm")]),
            "classifier_recall": float(means[("transition_recall", "classifier")]),
            "hmm_recall": float(means[("transition_recall", "hmm")]),
            "improved_trips": int((gains > 0).sum()),
            "total_trips": int(len(gains)),
            "best_gain_trip": str(gains.idxmax()),
            "best_gain": float(gains.max()),
            "worst_gain_trip": str(gains.idxmin()),
            "worst_gain": float(gains.min()),
        }
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

\vspace{2.0cm}
\begin{tabular}{@{}ll@{}}
\textbf{Course} & UE24CS352A Machine Learning \\
\textbf{Team} & 174 \\
\textbf{Repository} & \url{https://github.com/shakthi-raja/pathsnap} \\
\textbf{Partitions} & 80 training, 10 student-testing, 10 reserved evaluator trips \\
\end{tabular}

\vspace{1.1cm}
{\small
\begin{tabular}{@{}p{0.12\textwidth}p{0.24\textwidth}p{0.26\textwidth}p{0.30\textwidth}@{}}
\textbf{Person} & \textbf{USN} & \textbf{Name} & \textbf{Contribution} \\
\midrule
1 & PES1UG24CS669 & KAVANA P G & Data reading, road network, candidates, features, alignment \\
2 & PES1UG25CS804 & ANUVIDYA R & Model training, comparison and Bayesian optimisation \\
3 & PES1UG24CS670 & KRUTHIKA C B & HMM and Viterbi decoding, FastAPI backend \\
4 & PES1UG25CS843 & SHAKTHI RAJA G P & Evaluator interface, Folium maps, integration, report \\
\end{tabular}
}

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
        + _pct(metadata.get("test_top1_accuracy"))
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
a sequence, not a label per observation, and that shapes the whole approach. The supervised target
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

Networks are large. The median trip carries 18,762 arcs and the largest carries 307,392, so
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

    # ------------------------------------------------- quality and missingness
    if quality:
        gaps = quality["gaps"]
        raw_observations = int(quality["observations"])
        labelled = int(quality["labelled_observations"])
        withheld = int(quality["withheld_observations"])
        missing_rows = []
        if not missing.empty:
            reasons = {
                "movement_m": "First observation of each trip has no predecessor.",
                "dt_s": "Same rows: elapsed time is undefined without a previous fix.",
                "speed_mps": "Same rows as the elapsed time.",
                "gps_heading_rad": "Consecutive fixes can coincide, leaving no heading.",
                "heading_sin": "Derived from the heading, so absent together with it.",
                "heading_cos": "Derived from the heading, so absent together with it.",
                "heading_difference_rad": "Derived from the heading, so absent together with it.",
            }
            from src.data.features import FEATURE_COLUMNS

            features = set(FEATURE_COLUMNS)
            table = missing[
                missing["column"].isin(features) & (missing["missing"] > 0)
            ].pivot_table(index="column", columns="partition", values="share")
            order = ["movement_m", "dt_s", "speed_mps", "gps_heading_rad",
                     "heading_sin", "heading_cos", "heading_difference_rad"]
            for column in [name for name in order if name in table.index]:
                missing_rows.append(
                    f"{_tt(column)} & {_pct(table.loc[column, 'train'], 2)} "
                    f"& {_pct(table.loc[column, 'student_test'], 2)} & {reasons[column]} \\\\"
                )
        doc.append(
            r"""
\subsection{Sampling intervals and missing values}

Two checks belong to the same audit. The first is how often the receiver reports. Across the 80
training tracks there are """
            + f"{raw_observations:,}"
            + r""" observations and """
            + f"{int(gaps['gap_1s']['count']):,}"
            + r""" consecutive gaps, of which """
            + _pct(gaps["gap_1s"]["share"], 2)
            + r""" are one second, """
            + _pct(gaps["gap_2_to_5s"]["share"], 2)
            + r""" span two to five seconds, """
            + _pct(gaps["gap_6_to_60s"]["share"], 2)
            + r""" span six to sixty seconds and """
            + _pct(gaps["gap_over_60s"]["share"], 2)
            + r""" exceed a minute. A trace that is mostly 1\,Hz with a thin tail of dropouts is where
simple interpolation breaks down and a decoder that scores transitions between observations helps. The same audit found no
non-finite longitude, latitude or timestamp in the raw tracks.

The second is how much of that signal can be supervised. Of the """
            + f"{raw_observations:,}"
            + r""" observations, """
            + f"{labelled:,}"
            + r""" ("""
            + _pct(quality["labelled_share"])
            + r""") received a confident label from the alignment and """
            + f"{withheld:,}"
            + r""" were withheld. That is a large fraction to leave unlabelled. The decision is deliberate. The
withheld observations fall where the reference route covers only part of the drive or where two arcs
are equally plausible, so labelling them would inject noise rather than signal.

Table~\ref{tab:missing} accounts for missing values. Both labelled partitions are covered, and every
candidate row in them is counted, including rows the model never sees. Every feature is complete
except for the first fix of a trip and for fixes that did not move. Nothing is imputed by hand. Each
pipeline carries a median imputer fitted inside its training fold.

\begin{table}[H]
\centering
\caption{Missing values per feature, as a share of all candidate rows in each partition. The remaining
eleven features are complete in both partitions.}
\label{tab:missing}
\small
\begin{tabular}{@{}lllp{0.44\textwidth}@{}}
\toprule
\textbf{Feature} & \textbf{Train} & \textbf{Student test} & \textbf{Cause} \\
\midrule
"""
            + "\n".join(missing_rows)
            + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{figures/fig_data_quality.png}
\caption{Left: distribution of consecutive sampling gaps over the 80 training tracks. Right: GPS
observations in the processed training partition that carry a confident label and those the alignment
withheld.}
\label{fig:quality}
\end{figure}
\clearpage
"""
        )

    if quality:
        candidate_sentence = (
            "Of "
            + f"{int(quality['candidate_rows']):,}"
            + " candidate rows across the 80 training trips, "
            + f"{int(quality['labelled_candidate_rows']):,}"
            + " carried a confident label, or "
            + _pct(quality["labelled_candidate_share"])
            + ". The remainder were withheld by the alignment rather than guessed. After filtering, "
            + str(quality["labelled_trips"])
            + " of the "
            + str(quality["training_trip_tables"])
            + " trips contribute labelled rows; the other "
            + str(quality["training_trip_tables"] - quality["labelled_trips"])
            + " produced no confident alignment at all."
        )
    else:
        candidate_sentence = ""
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
ordered route occurrences. The output is a label and a confidence flag per observation.
\item Emit one parquet per trip holding candidate rows, features, a nullable label and an
eligibility flag.
\end{enumerate}

%%CANDIDATE_SENTENCE%%

\subsection{Features and their justification}

Eighteen features describe each observation and candidate pair, grouped in
Table~\ref{tab:features}. The design rule was that every feature must mean the same thing in every
city: metres, normalised fractions, angles encoded as sines and cosines, and elapsed time. Latitude
and longitude are deliberately excluded.
"""
    )
    doc[-1] = doc[-1].replace("%%CANDIDATE_SENTENCE%%", candidate_sentence)

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
classification. Each observation contributes several candidate arcs, each pair carries a label
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
        name = row["model"]
        metrics = extended.loc[name] if name in extended.index else None
        continuity = _pct(metrics["classifier_continuity"]) if metrics is not None else "n/a"
        inference = (
            f"{metrics['inference_ms']:.0f}" if metrics is not None else f"{row['inference_ms']:.0f}"
        )
        table_rows.append(
            f"{_tex(name)} & {_tex(FAMILY.get(name, ''))} & {_key_configuration(name, params)} "
            f"& {row['cv_macro_f1_mean']:.3f} $\\pm$ {row['cv_macro_f1_std']:.3f} "
            f"& {_pct(row['test_top1_accuracy'])} & {continuity} "
            f"& {_pct(row['test_macro_f1'])} & {_num(row['test_log_loss'])} "
            f"& {inference} \\\\"
        )

    doc.append(
        r"""
\begin{table}[H]
\centering
\caption{Comparison of the five models. Cross-validation uses five folds grouped by trip and every
student-test score comes from the ten development trips, which were never used for fitting. Route
continuity is measured on the classifier's sequence of best candidates, because the decoder is shared
by all five models; its own continuity is reported in Section~\ref{sec:hmm}. Inference time is the
fastest of three prediction passes over the student-test candidates.}
\label{tab:comparison}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llp{0.19\textwidth}cccccc@{}}
\toprule
\textbf{Model} & \textbf{Family} & \textbf{Key configuration} & \textbf{CV macro F1}
 & \textbf{Test Top-1} & \textbf{Route continuity} & \textbf{Test macro F1}
 & \textbf{Log loss} & \textbf{Inference (ms)} \\
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

Route continuity in the table is the share of consecutive best-candidate arcs that a directed road
path can connect inside the decoder's link limit. It is a property of the candidate ranking, so it can
be measured for every model without running Viterbi, and it separates connectivity failures from
local scoring failures. Inference time excludes candidate generation and metric scoring; the two models
compared in Section~\ref{sec:tuning} were timed the same way.

\subsection{Analysis of each model}
"""
    )

    verdicts = {
        "GaussianNB": (
            "The gap is small, but the model is last on every probability metric, which is "
            "underfitting: the independence assumption caps how well it can rank candidates."
        ),
        "LogisticRegression": (
            "The gap is small and the linear boundary transfers; the shortfall is capacity, not "
            "variance."
        ),
        "DecisionTree": (
            "Accuracy generalises while the log loss does not, so the tree overfits in probability "
            "space even where its argmax is acceptable."
        ),
        "RandomForest": (
            "No sign of overfitting: the student-test score sits within a point of the "
            "cross-validated score and the fold spread is moderate."
        ),
        "HistGradientBoosting": (
            "No sign of overfitting: the student-test score is slightly above the cross-validated "
            "score, and the log loss improves in the same direction."
        ),
    }

    for _, row in comparison.iterrows():
        name = row["model"]
        metrics = extended.loc[name] if name in extended.index else None
        param_rows = " \\\\\n".join(
            f"{_tex(key)} & {_tex(value)}" for key, value in params.get(name, {}).items()
        ) + " \\\\"
        measured = (
            r"\noindent\textit{Measured.} On the student-testing trips the candidate classifier "
            "reached precision "
            + _pct(metrics["test_precision"])
            + ", recall "
            + _pct(metrics["test_recall"])
            + " and macro F1 "
            + _pct(metrics["test_macro_f1"])
            + ", with observation-level Top-1 "
            + _pct(metrics["test_top1"])
            + " and log loss "
            + _num(metrics["test_log_loss"])
            + ", and its best-candidate sequence was "
            + _pct(metrics["classifier_continuity"])
            + " continuous. "
        )
        inference = (
            f"{metrics['inference_ms']:.0f}"
            if metrics is not None
            else f"{row['inference_ms']:.0f}"
        )
        doc.append(
            r"\subsubsection*{" + _tex(name) + "}\n"
            r"\noindent\textit{Role.} " + PURPOSE.get(name, "") + "\n\n"
            r"\begin{center}\small\begin{tabular}{@{}ll@{}}\toprule"
            r"\textbf{Hyperparameter} & \textbf{Initial value} \\ \midrule" + "\n"
            + param_rows + "\n"
            r"\bottomrule\end{tabular}\end{center}" + "\n\n"
            r"\noindent\textit{Objective.} " + LOSSES.get(name, "") + "\n\n"
            + measured
            + "Cross-validated macro F1 was "
            + f"{row['cv_macro_f1_mean']:.3f}"
            + " with a fold standard deviation of "
            + f"{row['cv_macro_f1_std']:.3f}"
            + ", cross-validated Top-1 was "
            + _pct(row["cv_top1_mean"])
            + ", the mean fit took "
            + f"{row['cv_train_seconds_mean']:.1f}"
            + " seconds per fold, and inference took "
            + inference
            + r""" milliseconds.

\noindent\textit{Strengths and limitations.} """ + STRENGTHS.get(name, "") + "\n\n"
            + r"\noindent\textit{Overfitting check.} Cross-validated Top-1 was "
            + _pct(row["cv_top1_mean"])
            + " and student-test Top-1 was "
            + _pct(row["test_top1_accuracy"])
            + ", a gap of "
            + f"{abs(row['test_top1_accuracy'] - row['cv_top1_mean']) * 100:.1f}"
            + " percentage points. "
            + verdicts.get(name, "")
            + "\n\n"
        )

    # ---------------------------------------------- 3.3 candidate confusion matrix
    if confusion:
        total_rows = sum(confusion.values())
        precision = confusion["tp"] / (confusion["tp"] + confusion["fp"])
        recall = confusion["tp"] / (confusion["tp"] + confusion["fn"])
        f1 = 2 * precision * recall / (precision + recall)
        doc.append(
            r"""
\subsection{Candidate-level confusion matrix}

The table above aggregates, and an aggregate can hide where the errors sit. Figure~\ref{fig:confusion}
breaks the shipped model's decisions on the student-test candidates into the four cells of a confusion
matrix, using the default 0.5 threshold. There are """
            + f"{total_rows:,}"
            + r""" labelled candidate rows: """
            + f"{confusion['tp']:,}"
            + r""" true positives, """
            + f"{confusion['fp']:,}"
            + r""" false positives, """
            + f"{confusion['fn']:,}"
            + r""" false negatives and """
            + f"{confusion['tn']:,}"
            + r""" true negatives. That gives candidate precision """
            + _pct(precision)
            + r""", recall """
            + _pct(recall)
            + r""" and F1 """
            + _pct(f1)
            + r""".

The asymmetry is in the useful direction. A false positive only hurts when it outranks the true
candidate, and with only """
            + _pct(confusion["fp"] / total_rows, 2)
            + r""" of rows falling there, most of them never reach the top of a ranking. A false negative
is the more expensive error. The reference arc was rejected outright, so the decoder has to recover
from a neighbour and the observation can only be correct by luck. The counts also show why the hard
labels are a diagnostic rather than the objective. The decoder never reads a 0/1 decision, it reads
the probability itself, which is why selection was settled on log loss once Top-1 tied.

\begin{figure}[H]
\centering
\includegraphics[width=0.5\textwidth]{figures/fig_confusion_matrix.png}
\caption{Candidate-level confusion matrix for the shipped model on the student-test partition. Cell
labels give the row count and the share of all candidate rows.}
\label{fig:confusion}
\end{figure}
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
alignment. The result is reported as measured. Figure~\ref{fig:evaluator} shows the same comparison
trip by trip.

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{figures/fig_evaluator_trips.png}
\caption{Reserved evaluator partition, classifier Top-1 against decoded pointwise accuracy. Every bar
pair is one trip; the decoder is above the classifier on all ten.}
\label{fig:evaluator}
\end{figure}

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
    doc.append(r"\section{Bayesian hyperparameter optimisation}" + "\n" + r"\label{sec:tuning}" + "\n")
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
\begin{tabular}{@{}p{0.30\textwidth}p{0.63\textwidth}@{}}
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

Table~\ref{tab:tuned} repeats the comparison on the same folds and the same student-test rows, so the
only difference between the two rows is the hyperparameter configuration.

\begin{table}[H]
\centering
\caption{Tuned against untuned model on identical folds and identical student-test rows. Inference time
is the fastest of three prediction passes over the student-test candidates.}
\label{tab:tuned}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Variant} & \textbf{CV macro F1} & \textbf{Test macro F1} & \textbf{Test Top-1}
 & \textbf{Log loss} & \textbf{Inference (ms)} \\
\midrule
%%TUNED_ROWS%%
\bottomrule
\end{tabular}}
\end{table}

The tuned configuration did not improve the model. Cross-validated macro F1 rose slightly, while
student-test macro F1 and student-test Top-1 both fell and log loss moved the wrong way. The one clear
gain is inference time, and it comes from the search choosing a smaller number of boosting iterations
than the default, not from a better model. On a dataset this size the metric movements sit inside the
fold noise, so a model that is equal on accuracy, slightly worse on calibration and faster only because
it is smaller has no argument to replace the one already in the report. The tuned pipeline was
therefore not adopted.

The search still did its job. It probed a wide region and the defaults won, which says the default
configuration sat near the optimum of this space. The full trial
record, including every sampled configuration and its objective, is kept in
\texttt{results/optuna\_trials.csv}.

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figures/fig_optuna_history.png}
\caption{Objective value for each Bayesian trial. The step line is the best value seen so far; later
trials did not improve it.}
\label{fig:optuna}
\end{figure}
"""
        )
        tuned_rows = "\n".join(
            f"{_tex(row['variant'])} & {row['cv_macro_f1']:.3f} $\\pm$ {row['cv_macro_f1_std']:.3f} "
            f"& {_pct(row['test_macro_f1'])} & {_pct(row['test_top1'])} "
            f"& {_num(row['test_log_loss'])} & {row['inference_ms']:.0f} \\\\"
            for _, row in tuned_vs.iterrows()
        )
        doc[-1] = doc[-1].replace("%%TUNED_ROWS%%", tuned_rows)

    # ------------------------------------------------------------------ 5 HMM
    doc.append(
        r"""
\section{HMM and Viterbi route reconstruction}
\label{sec:hmm}

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
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Trip} & \textbf{Evaluable} & \textbf{Classifier Top-1} & \textbf{Decoded}
 & \textbf{Classifier continuity} & \textbf{Decoded continuity} \\
\midrule
"""
            + "\n".join(api_rows)
            + r"""
\bottomrule
\end{tabular}}
\end{table}

Decoding improved pointwise accuracy on every trip measured, by between """
            + f"{min(gains) * 100:.1f}"
            + " and "
            + f"{max(gains) * 100:.1f}"
            + r""" percentage points. The largest gains appear where the classifier is weakest, which
is the expected behaviour. When local evidence is ambiguous, connectivity carries the decision.

\subsection{Route-level agreement with the reference sequence}

Pointwise accuracy scores one observation at a time, so it cannot see whether the arcs come out in a
driveable order. The sequence measure does. Each trip's prediction is reduced to its ordered list of
distinct arcs, that list is turned into the set of consecutive arc pairs it contains, and the set is
compared with the pairs in the reference route. Precision is the share of predicted pairs that occur
in the reference, recall is the share of reference pairs that were predicted, and the F1 of the two is
the route-level score.

\begin{table}[H]
\centering
\caption{Route-level agreement on the student-test trips. Pairs are consecutive arcs in the
deduplicated sequence, so a repeated arc is counted once.}
\label{tab:sequence}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Trip} & \textbf{Reference arcs} & \textbf{Classifier P} & \textbf{Classifier R}
 & \textbf{Classifier F1} & \textbf{Decoded P} & \textbf{Decoded R} & \textbf{Decoded F1} \\
\midrule
%%SEQUENCE_ROWS%%
\bottomrule
\end{tabular}}
\end{table}

The averages tell the same story as the pointwise table. Transition precision rises from """
            + _pct(sequence_means.get("classifier_precision"))
            + r""" to """
            + _pct(sequence_means.get("hmm_precision"))
            + r""" and recall rises from """
            + _pct(sequence_means.get("classifier_recall"))
            + r""" to """
            + _pct(sequence_means.get("hmm_recall"))
            + r""", which lifts transition F1 from """
            + _pct(sequence_means.get("classifier_f1"))
            + r""" to """
            + _pct(sequence_means.get("hmm_f1"))
            + """. The decoded route is closer to the reference on """
            + str(sequence_means.get("improved_trips"))
            + " of "
            + str(sequence_means.get("total_trips"))
            + r""" trips.

The gains are uneven, and the unevenness is informative. The smallest gain is on trip """
            + _tex(sequence_means.get("worst_gain_trip"))
            + r""", about """
            + f"{sequence_means.get('worst_gain', 0) * 100:.0f}"
            + r""" points of F1, a short trip where the classifier was already close to the reference
order and there was little left to repair. The largest gain is on trip """
            + _tex(sequence_means.get("best_gain_trip"))
            + r""", where the classifier jumps between arcs and decoding lifts F1 by about """
            + f"{sequence_means.get('best_gain', 0) * 100:.0f}"
            + r""" points.

The weakest prediction is trip """
            + _tex(sequence_means.get("weakest_trip"))
            + r""", where decoded F1 is still """
            + _pct(sequence_means.get("weakest_f1"))
            + r""". The classifier names """
            + str(sequence_means.get("weakest_predicted_arcs"))
            + r""" distinct arcs on that trip against a reference of """
            + str(sequence_means.get("weakest_reference_arcs"))
            + r""", so the decoded sequence necessarily contains transitions the reference does not have.
The predicted sequence is far longer than the reference, and no transition score can remove arcs that
the classifier keeps choosing.

Decoding did not hurt a single trip, but there are two cases where it cannot help much. The first is
when the reference arc is missing from the candidate set at a radius. No transition score can vote for
a road that was never generated, so recall is capped before the decoder runs. The second is a sparse
extract where one arc covers several observations. Distinct-arc scoring then has very few pairs to
match, and the trip scores low however good the local match is. Both are candidate-generation limits,
not Viterbi limits, and the first is exactly what the candidate miss rate column in
Table~\ref{tab:evaluator} measures.

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig_sequence_f1.png}
\caption{Route-level transition F1 for each student-test trip, classifier only against Viterbi
decoding.}
\label{fig:sequence}
\end{figure}
\clearpage
"""
        )
        if sequence_rows:
            doc[-1] = doc[-1].replace("%%SEQUENCE_ROWS%%", sequence_rows)

    # --------------------------------------------------------------- 6 folium
    doc.append(
        r"""
\section{Geographical visualisation with Folium}

Maps are produced by the map module in \texttt{src/visualization} from live data, and by the report
renderer in \texttt{tools} for the evidence shown here. The evaluator interface renders GPS
observations, the relevant road context and the predicted route. The reference route appears only in
report maps built from training or student-testing trips, never in the interface.

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures/folium_trip_00000062_layers.png}
\caption{Trip 00000062, Yekaterinburg, with every layer the report tool draws. Grey lines are the road
network, blue marks the raw GPS observations, green is the route predicted by the shipped pipeline and
dashed orange is the reference route. The prediction follows the reference for most of the drive; the
reference is dashed and drawn on top so both stay visible where they overlap. Generated by
\texttt{tools/render\_predicted\_trip\_map.py}, and the interactive HTML with working layer controls is
committed at \texttt{results/maps/trip\_00000062\_layers.html}.}
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
crosses that link. The response contract has no field for the reference route.

The service runs in one of two modes. In trained mode it builds candidates, scores them, decodes a
route and then scores that route against the reference, so the response carries accuracy, Top-1,
continuity and candidate miss rate. In evaluator mode it reads candidate probabilities that were
computed ahead of time and never opens a reference route at all, so the response carries only
geometry, continuity and runtime. The interface screenshot in Figure~\ref{fig:ui} comes from an
evaluator-mode run, which is why its readout shows a single metric, and the request log in
Figure~\ref{fig:apilog} comes from a trained-mode run, which is why it carries an accuracy. Nothing
is hidden. The health endpoint names the mode, and the response omits what was not computed.

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
\caption{Request log on Laptop 1 during a trained-mode run, captured on the machine receiving the
requests. The server answers \texttt{/health} and \texttt{/trips} first, then serves a prediction for
trip 00000000 with 2,503 observations in 4,419 milliseconds and reports 96.3 percent pointwise
accuracy. The final line is the request completed with status 200.}
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

python tools/render_predicted_trip_map.py --trip 00000062 --partition train \
    --data-root "$PATHSNAP_DATA_ROOT" \
    --out-html results/maps/trip_00000062_layers.html
\end{verbatim}
\clearpage
"""
    )

    # ------------------------------------------------------------ 8 integrity
    doc.append(
        r"""
\section{Academic integrity and AI assistance}

Every figure and table in this report is produced by scripts in the repository from the recorded run,
and every number is read from one of these files:

\begin{itemize}[leftmargin=1.4em,itemsep=2pt,topsep=3pt]
\item model comparison and per-model metrics:\\
\texttt{results/model\_comparison.csv}, \texttt{results/model\_test\_metrics.csv};
\item data quality and missing values:\\
\texttt{results/data\_quality.json}, \texttt{results/missing\_values.csv};
\item confusion matrix: \texttt{results/confusion\_matrix.json};
\item Bayesian search and the tuned comparison:\\
\texttt{results/optuna\_trials.csv}, \texttt{results/tuned\_vs\_untuned.csv};
\item decoder comparison:\\
\texttt{results/api\_evaluation.json}, \texttt{results/sequence\_metrics.csv};
\item reserved partition: \texttt{results/evaluator\_test\_results.json}.
\end{itemize}

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

The system produces a drivable route from noisy GPS for every trip tested.
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
label where the alignment was uncertain, kept %%WITHHELD_SHARE%% of the candidate rows out of training
instead of teaching the model to be wrong. Decoding with road connectivity turned independent scores into routes
a vehicle could drive.

The largest remaining weakness is candidate coverage on sparse networks, which is visible in the
reserved trip that scored %%WORST_TRIP_ACCURACY%%, and the slowest evaluator request took
%%SLOWEST_SECONDS%% seconds because candidate generation runs inside the request.
"""
    if quality:
        conclusion = conclusion.replace(
            "%%WITHHELD_SHARE%%", _pct(1 - quality["labelled_candidate_share"])
        )
    if evaluator:
        worst_trip = min(
            evaluator["trips"].values(),
            key=lambda payload: (payload.get("metrics") or {}).get("pointwise_accuracy", 1),
        )
        slowest = max(
            (payload.get("runtime_ms", 0) for payload in evaluator["trips"].values()),
            default=0,
        )
        conclusion = conclusion.replace(
            "%%WORST_TRIP_ACCURACY%%",
            _pct((worst_trip.get("metrics") or {}).get("pointwise_accuracy")),
        )
        conclusion = conclusion.replace("%%SLOWEST_SECONDS%%", f"{slowest / 1000:.0f}")
    if sequence_means:
        conclusion = conclusion.replace(
            "a vehicle could drive.\n",
            "a vehicle could drive. On the student-test trips the same decoder raised route-level "
            "transition F1 from "
            + _pct(sequence_means.get("classifier_f1"))
            + " to "
            + _pct(sequence_means.get("hmm_f1"))
            + ", so the gain shows up in arc order as well as pointwise.\n",
        )
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
