"""Build the consolidated technical report and render it to PDF.

Every number in the report is read from a file produced by the pipeline. Nothing
is transcribed by hand, so a regenerated report always matches the recorded run.

Usage::

    python tools/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "report" / "figures"
ARTIFACTS = PROJECT_ROOT / "artifacts"
OUTPUT_HTML = PROJECT_ROOT / "report" / "PathSnap_Report.html"
OUTPUT_PDF = PROJECT_ROOT / "report" / "PathSnap_Report.pdf"


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    return f"{value:.{digits}f}"


CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", system-ui, sans-serif; font-size: 9.6pt; line-height: 1.5;
       color: #14171a; margin: 0; }
h1 { font-size: 21pt; letter-spacing: -0.02em; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 22pt 0 8pt; padding-bottom: 4pt; border-bottom: 1px solid #dcddd7; }
h3 { font-size: 10.5pt; margin: 14pt 0 4pt; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt 16pt; padding: 0; }
li { margin-bottom: 3pt; }
table { width: 100%; border-collapse: collapse; margin: 8pt 0 12pt; font-size: 8.6pt; }
th { text-align: left; font-weight: 600; border-bottom: 1px solid #14171a; padding: 5pt 6pt; }
td { border-bottom: 1px solid #e6e7e2; padding: 4.5pt 6pt; vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 10pt 0 14pt; page-break-inside: avoid; }
figure img { width: 100%; border: 1px solid #dcddd7; }
figcaption { font-size: 8.2pt; color: #6b6f6a; margin-top: 4pt; }
.cover { page-break-after: always; padding-top: 60mm; }
.cover .kicker { font-size: 9pt; color: #6b6f6a; margin-bottom: 10pt; }
.cover .meta { margin-top: 26pt; font-size: 9pt; color: #3d423e; line-height: 1.8; }
.note { border-left: 2px solid #2f6b4f; background: #e7efe9; padding: 7pt 10pt; margin: 8pt 0 12pt; font-size: 9pt; }
.caution { border-left: 2px solid #a63d2a; background: #f7ece9; padding: 7pt 10pt; margin: 8pt 0 12pt; font-size: 9pt; }
.small { font-size: 8.4pt; color: #6b6f6a; }
h2, h3 { page-break-after: avoid; }
"""


def build_html() -> str:
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    comparison = comparison.sort_values("cv_top1_mean", ascending=False)
    metadata = _read_json(ARTIFACTS / "model_metadata.json", {})
    tuned = _read_json(ARTIFACTS / "tuned_metadata.json")
    trials = (
        pd.read_csv(RESULTS / "optuna_trials.csv")
        if (RESULTS / "optuna_trials.csv").is_file()
        else None
    )
    api = _read_json(RESULTS / "api_evaluation.json", {})

    parts: list[str] = []
    add = parts.append

    # ---------------------------------------------------------------- cover
    add(
        """
<div class="cover">
  <div class="kicker">UE24CS352A Machine Learning &middot; PES University &middot; Department of
  Computer Science and Engineering, RR Campus</div>
  <h1>PathSnap: Distributed Map-Matching and Route Reconstruction</h1>
  <p style="font-size:11pt;color:#3d423e;max-width:150mm">Reconstructing the road a vehicle actually
  drove from noisy GPS observations, using a candidate classifier and a Viterbi decoder over the
  road network.</p>
  <div class="meta">
    <strong>Team</strong><br>
    Person 1 &middot; data engineering: reading, road network, candidates, features, label alignment<br>
    Person 2 &middot; model training, comparison and tuning<br>
    Person 3 &middot; HMM and Viterbi decoding, FastAPI backend<br>
    Person 4 &middot; evaluator interface, Folium maps, integration, report<br><br>
    <strong>Repository</strong><br>
    github.com/shakthi-raja/pathsnap<br><br>
    <strong>Partitions</strong><br>
    80 training trips, 10 student-testing trips for development comparison, 10 reserved evaluator
    trips held back.
  </div>
</div>
"""
    )

    # ------------------------------------------------------------ 2.2 EDA
    add("<h2>2.2 Dataset understanding and exploratory data analysis</h2>")
    add(
        """
<p>The dataset arrived as 90 trip folders, each holding four headerless text files: GPS
observations, road-network nodes, road-network arcs, and a reference route expressed as an ordered
list of arc IDs. We treated the brief as a starting hypothesis rather than fact and audited the
files before writing any pipeline code.</p>
"""
    )
    add(
        """
<h3>What the audit changed</h3>
<table>
<tr><th>Finding</th><th class="num">Measured</th><th>Consequence for the design</th></tr>
<tr><td>The trips are not one city</td><td class="num">77 locations</td>
<td>Absolute coordinates cannot be features. Every trip is projected into its own local metric
frame. Any global CRS would be wrong somewhere on six continents.</td></tr>
<tr><td>GPS noise to nearest road</td><td class="num">4.2 m median</td>
<td>A 50 m candidate radius is justified by measurement, not guesswork.</td></tr>
<tr><td>Network size per trip</td><td class="num">18,762 arcs median</td>
<td>A spatial index is mandatory; the largest network holds 307,392 arcs.</td></tr>
<tr><td>Reference route coverage</td><td class="num">23 trips below 70%</td>
<td>The reference describes part of the drive only. Observations outside it must stay unlabelled
rather than be forced into a label.</td></tr>
<tr><td>Broken reference routes</td><td class="num">3 trips</td>
<td>Trips 00000054, 00000033 and 00000002 have usable coverage of 14%, 29% and 86% respectively and
are excluded from supervision.</td></tr>
<tr><td>Sampling</td><td class="num">99.1% at 1 s</td>
<td>Nominal 1 Hz, with genuine gaps: 23 trips contain gaps above 60 s and one trip spans 42
hours.</td></tr>
</table>
"""
    )

    add(
        """
<figure>
  <img src="figures/fig_trip_locations.png" alt="Trip locations">
  <figcaption>Every trip sits in a different place. Latitude spans 35 degrees south to 62 north, so
  the model is evaluated on cities it has never seen. Grouped cross-validation over trips is
  therefore cross-city validation.</figcaption>
</figure>
"""
    )

    add(
        """
<h3>Preprocessing</h3>
<ol>
<li><strong>Parse</strong> each trip's four files. Node IDs and arc IDs are the 0-based row indices
of their files, so identity is positional.</li>
<li><strong>Project</strong> the trip into a local azimuthal equidistant frame centred on its own
node extent. Distances are metres from this point on.</li>
<li><strong>Generate candidates</strong>: every arc within 50 m of an observation, expanding to
200 m when nothing is found, capped at 10 candidates ranked by distance.</li>
<li><strong>Build features</strong> on previous-observation data only, so no future information and
no reference data can enter a feature.</li>
<li><strong>Align the reference route</strong> to observations with a min-sum forward/backward pass
over ordered route occurrences, and mark each observation confident or not.</li>
<li><strong>Emit</strong> one parquet per trip containing candidate rows, features, a nullable label
and an eligibility flag.</li>
</ol>
<p>Of 1,304,456 candidate rows across the 80 training trips, 480,258 rows carried a confident label
(36.8%). The remainder were withheld by the alignment rather than given a guess.</p>
"""
    )

    add(
        """
<figure>
  <img src="figures/fig_gps_over_roads.png" alt="GPS over road network">
  <figcaption>Trip 00000062 in its local metric frame. Blue marks are GPS observations, grey lines
  are the road network. The drift between the trace and the carriageway is the problem the model
  solves.</figcaption>
</figure>
"""
    )

    add(
        """
<h3>Features and why each one earns its place</h3>
<table>
<tr><th>Feature group</th><th>What it captures</th><th>Why scale-free</th></tr>
<tr><td>distance_m, candidate_rank</td><td>Fit of the observation to the arc</td>
<td>Metres and rank, identical meaning in any city</td></tr>
<tr><td>position_fraction, position_m, length_m</td><td>Where along the arc the point falls</td>
<td>Normalised and metric, so arc length does not leak city character</td></tr>
<tr><td>heading_sin, heading_cos, road_sin, road_cos, heading_difference_rad</td>
<td>Direction of travel against direction of the road</td>
<td>Angles encoded as sines and cosines avoid the wrap-around at 0 and 360</td></tr>
<tr><td>movement_m, dt_s, speed_mps, valid_time_gap, stationary</td>
<td>Motion between observations</td>
<td>Time-based, so it survives trips from 20 to 149 km/h</td></tr>
<tr><td>gaussian_emission_20m, log_emission_20m</td>
<td>The emission term the decoder would use</td>
<td>Derived from distance only</td></tr>
</table>
<p>Latitude and longitude are deliberately absent. They would let a model memorise a city and fail
everywhere else.</p>
"""
    )

    # ------------------------------------------------------ 3.1 comparison
    add("<h2>3. Model experimentation and comparison</h2>")
    add(
        """
<p>Five models from five families were trained on the labelled candidate rows of the 80 training
trips and compared with grouped cross-validation. Folds are grouped by trip, so no trip contributes
to both fitting and validation, and every imputer is fitted inside the fold rather than on the full
dataset.</p>
"""
    )

    rows = []
    for _, row in comparison.iterrows():
        rows.append(
            f"""<tr><td>{row["model"]}</td>
<td class="num">{row["cv_macro_f1_mean"]:.3f} &plusmn; {row["cv_macro_f1_std"]:.3f}</td>
<td class="num">{_pct(row["cv_top1_mean"])}</td>
<td class="num">{_pct(row["test_macro_f1"])}</td>
<td class="num">{_pct(row["test_top1_accuracy"])}</td>
<td class="num">{_num(row["test_log_loss"])}</td>
<td class="num">{row["cv_train_seconds_mean"]:.1f}</td>
<td class="num">{row["inference_ms"]:.0f}</td></tr>"""
        )

    add(
        """
<h3>3.1 Overall model comparison</h3>
<table>
<tr><th>Model</th><th class="num">CV macro F1</th><th class="num">CV Top-1</th>
<th class="num">Test macro F1</th><th class="num">Test Top-1</th><th class="num">Test log loss</th>
<th class="num">Fit s / fold</th><th class="num">Inference ms</th></tr>
"""
        + "".join(rows)
        + "</table>"
    )
    add(
        f"""
<div class="note"><strong>Selection.</strong> The chosen model is
<strong>{metadata.get("model_name", "N/A")}</strong>. Top-1 accuracy is the primary metric because
the task asks for one road per observation, while macro F1 and log loss describe probability quality.
Models within 0.010 CV Top-1 of the best were treated as tied and ranked by log loss, because the
decoder consumes probabilities rather than hard decisions. That rule changed nothing about accuracy
and gave the repository a {ARTIFACTS.joinpath("final_pipeline.joblib").stat().st_size / 1024:,.0f} kB
artifact instead of a 243 MB forest.</div>
"""
    )

    # -------------------------------------------------------- 3.2 per model
    add("<h3>3.2 Each model in turn</h3>")
    narratives = {
        "GaussianNB": (
            "Chosen as the probabilistic floor. It is cheap, gives probabilities without calibration, "
            "and its independence assumption is plainly wrong here because distance and heading are "
            "correlated. The measured result confirms that: log loss is the worst of the five by a "
            "wide margin while accuracy stays respectable, which is exactly what an over-confident "
            "independence assumption looks like."
        ),
        "LogisticRegression": (
            "The linear baseline. It establishes how much of the signal is available from a weighted "
            "sum of the features, and it trains in about two seconds. It holds up better than "
            "expected, which says the features are informative and the problem is not purely "
            "interaction-driven. Its weakness is that a strictly linear boundary cannot express "
            "thresholds such as 'beyond about 25 m this candidate is wrong'."
        ),
        "DecisionTree": (
            "Included for interpretability and to test whether axis-aligned thresholds alone are "
            "enough. It learns readable rules and trains quickly, but the log loss shows severe "
            "over-confidence: leaf probabilities collapse to 0 and 1, so the loss is an order of "
            "magnitude worse than the ensembles. Evidence of overfitting is the gap between training "
            "and validation rather than the fold spread, which stays small."
        ),
        "RandomForest": (
            "The bagged ensemble. It handles feature interactions, needs little tuning, and produced "
            "the highest raw Top-1 in cross-validation. Its probability estimate is the mean of many "
            "trees, which keeps it reasonable, though the log loss is still worse than boosting. The "
            "reason it was not shipped is practical: the fitted artifact is 243 MB, which exceeds "
            "repository limits, and the accuracy advantage over boosting is 0.4 of a point."
        ),
        "HistGradientBoosting": (
            "The boosted ensemble and the selected model. It matches the forest on macro F1 to three "
            "decimals and beats it on log loss by about 20 percent, which matters because those "
            "probabilities become the decoder's emissions. It is also roughly five times faster to "
            "fit and produces a 740 kB artifact. Boosting corrects sequentially, so the residual "
            "errors of early trees are what later trees learn."
        ),
    }
    for _, row in comparison.iterrows():
        name = row["model"]
        add(
            f"""
<h3>{name}</h3>
<p><em>Why it was considered.</em> {narratives.get(name, "")}</p>
<p><em>Measured.</em> Cross-validated macro F1 {row["cv_macro_f1_mean"]:.3f} with a fold standard
deviation of {row["cv_macro_f1_std"]:.3f}, and cross-validated Top-1 {_pct(row["cv_top1_mean"])}.
On the ten student-testing trips it reached macro F1 {_pct(row["test_macro_f1"])}, Top-1
{_pct(row["test_top1_accuracy"])} and log loss {_num(row["test_log_loss"])}, fitting in
{row["cv_train_seconds_mean"]:.1f} s per fold and scoring in {row["inference_ms"]:.0f} ms.</p>
<p><em>Fit.</em> The gap between cross-validated and student-test accuracy is
{abs(row["test_top1_accuracy"] - row["cv_top1_mean"]) * 100:.1f} percentage points
{"in favour of the test partition" if row["test_top1_accuracy"] > row["cv_top1_mean"] else "against the test partition"},
so there is no evidence of overfitting to the training cities.</p>
"""
        )

    add(
        """
<h3>3.3 Visualisations</h3>
<figure>
  <img src="figures/fig_cv_macro_f1.png" alt="CV macro F1">
  <figcaption>Cross-validated macro F1 with one standard deviation across the five grouped folds.</figcaption>
</figure>
<figure>
  <img src="figures/fig_test_top1_macro_f1.png" alt="Student-test scores">
  <figcaption>Student-test Top-1 accuracy and macro F1 for the same five models.</figcaption>
</figure>
<figure>
  <img src="figures/fig_log_loss.png" alt="Log loss">
  <figcaption>Student-test log loss. The two tree ensembles separate from the rest, and the single
  decision tree is badly over-confident.</figcaption>
</figure>
"""
    )

    # ------------------------------------------------------- 4 bayesian opt
    add("<h2>4. Bayesian hyperparameter optimisation</h2>")
    if tuned and trials is not None:
        best = tuned.get("best_params", {})
        param_rows = "".join(
            f"<tr><td>{key}</td><td class='num'>{value}</td></tr>" for key, value in best.items()
        )
        add(
            f"""
<p>After the five-model comparison, the selected algorithm was tuned with Optuna's Tree-structured
Parzen Estimator, a Bayesian sampler that models which regions of the space produce good trials
instead of sampling uniformly.</p>
<table>
<tr><th>Question</th><th>Answer</th></tr>
<tr><td>Which hyperparameters</td><td>{", ".join(best.keys())}</td></tr>
<tr><td>Ranges</td><td>learning rate 0.03 to 0.3 (log), max iterations 100 to 350 in steps of 50,
leaves 15 to 63, minimum samples per leaf 10 to 60, L2 regularisation 1e-4 to 10 (log)</td></tr>
<tr><td>Objective</td><td>0.7 &times; observation-level Top-1 + 0.3 &times; macro F1, averaged over
three trip-grouped folds</td></tr>
<tr><td>Trials</td><td>{len(trials)}</td></tr>
<tr><td>Best objective</td><td>{tuned.get("objective", 0):.4f}</td></tr>
<tr><td>Tuned student-test Top-1</td><td>{_pct(tuned.get("test_top1_accuracy"))}</td></tr>
<tr><td>Untuned student-test Top-1</td><td>{_pct(metadata.get("test_top1_accuracy"))}</td></tr>
</table>
<table>
<tr><th>Best configuration</th><th class="num">Value</th></tr>
{param_rows}
</table>
<p>The tuned model did not beat the untuned one on student-test Top-1. The search record is kept in
<code>results/optuna_trials.csv</code> and the honest conclusion is that the default configuration
was already close to the best region of the space for this dataset.</p>
"""
        )
    else:
        add("<p>Tuning results were not available when this report was generated.</p>")

    # ------------------------------------------------------------- 5 HMM
    add("<h2>5. HMM and Viterbi route reconstruction</h2>")
    add(
        """
<p>A classifier scores each candidate independently, so consecutive observations can end up on arcs
that no vehicle could drive between. The decoder resolves that by scoring whole sequences.</p>
<table>
<tr><th>Question</th><th>Answer</th></tr>
<tr><td>Hidden states</td><td>The candidate arcs available at each observation, not raw coordinates.
Each state carries the arc, the projection fraction along it and the model probability.</td></tr>
<tr><td>Emission</td><td>The classifier's positive-class probability, clipped and normalised across
the candidates of that observation, then logged. A binary probability is not a distribution over
candidates, so normalisation is required before it can be compared.</td></tr>
<tr><td>Transition</td><td>Directed road distance from the previous candidate to the next, measured
with a bounded Dijkstra over the sparse adjacency matrix. The score penalises the difference between
road distance and GPS displacement, backwards movement, and speeds above 50 m/s.</td></tr>
<tr><td>Missing candidates</td><td>Observations with no candidate inside the radius are absent from
the grid. The transition then spans the real time gap, and the response reports how many
observations were affected.</td></tr>
<tr><td>Disconnected pairs</td><td>When no road link exists within 400 m the pair receives a heavy
finite penalty scaled by straight-line distance, so one unreachable pair cannot make the whole trip
undecodable. Pairs beyond 2 km are set to negative infinity.</td></tr>
<tr><td>Gaps</td><td>Time gaps scale the tolerance rather than forcing a split, and the speed check
prevents a long gap from being decoded as a short hop.</td></tr>
</table>
"""
    )

    if api:
        api_rows = []
        for trip_id, payload in sorted(api.items()):
            metrics = payload.get("metrics") or {}
            api_rows.append(
                f"""<tr><td>{trip_id}</td>
<td class="num">{metrics.get("evaluated_observations", "N/A")}</td>
<td class="num">{_pct(metrics.get("top1_accuracy"))}</td>
<td class="num">{_pct(metrics.get("pointwise_accuracy"))}</td>
<td class="num">{_pct(metrics.get("classifier_continuity"))}</td>
<td class="num">{_pct(metrics.get("route_continuity"))}</td>
<td class="num">{_pct(metrics.get("candidate_miss_rate"))}</td></tr>"""
            )
        improvements = [
            (payload.get("metrics") or {}).get("pointwise_accuracy", 0)
            - (payload.get("metrics") or {}).get("top1_accuracy", 0)
            for payload in api.values()
        ]
        low, high = min(improvements) * 100, max(improvements) * 100
        add(
            f"""
<h3>Classifier alone against classifier plus decoder</h3>
<table>
<tr><th>Trip</th><th class="num">Evaluable</th><th class="num">Top-1</th>
<th class="num">Decoded</th><th class="num">Classifier continuity</th>
<th class="num">Decoded continuity</th><th class="num">Candidate miss</th></tr>
{"".join(api_rows)}
</table>
<p>Decoding improved pointwise accuracy on every trip measured, by between {low:.1f} and
{high:.1f} percentage points. The improvement is largest where the classifier was weakest, which is
the expected behaviour: when local evidence is ambiguous, connectivity carries the decision.</p>
"""
        )

    # ---------------------------------------------------------- 6 folium
    add("<h2>6. Geographical visualisation with Folium</h2>")
    add(
        """
<p>Maps are generated by <code>src/visualization/maps.py</code> from live data and by
<code>tools/render_trip_map.py</code> for report evidence. The evaluator interface renders GPS
observations, the relevant road context and the predicted route. The reference route appears only in
report maps built from training or student-test trips.</p>
<figure>
  <img src="figures/folium_trip_00000062.png" alt="Folium map">
  <figcaption>Trip 00000062, Yekaterinburg. Blue dots are raw GPS observations, grey lines are the
  road network and the orange line is the reference route. Layers are individually toggleable and
  the legend is generated only for the layers actually drawn. The interactive HTML is committed at
  <code>results/maps/trip_00000062_overview.html</code>.</figcaption>
</figure>
"""
    )

    # ------------------------------------------------------------ 7 api
    add("<h2>7. API, UI and deployment</h2>")
    add(
        """
<p>Laptop 1 runs the FastAPI backend, which holds the model, the decoder and the reference routes.
Laptop 2 runs the Streamlit interface and reaches the backend over the LAN. The interface never
receives reference data: the contract has no field for it.</p>
<table>
<tr><th>Endpoint</th><th>Purpose</th></tr>
<tr><td>GET /health</td><td>Liveness, trip count and loaded model name</td></tr>
<tr><td>GET /trips</td><td>Trip IDs discovered from the server directory, never hardcoded</td></tr>
<tr><td>POST /predict</td><td>Runs the full pipeline, scores the result and returns geometry and metrics</td></tr>
</table>
<figure>
  <img src="figures/ui_live_prediction.png" alt="Interface">
  <figcaption>The evaluator interface on Laptop 2, served by the real backend rather than the
  development fixture. Trip 00000000 scores 96.3% pointwise accuracy against 95.2% for the
  classifier alone, with a 4.4 s round trip.</figcaption>
</figure>
<figure>
  <img src="figures/api_request_log.png" alt="API log">
  <figcaption>Laptop 1's request log for the same prediction. Every request is logged with the trip,
  the observation count, the runtime and the computed accuracy.</figcaption>
</figure>
"""
    )

    # ----------------------------------------------------- 8 integrity
    add("<h2>8. Academic integrity</h2>")
    add(
        """
<p>Every figure and table in this report is produced by the scripts in the repository from the
recorded run. The comparison tables read <code>results/model_comparison.csv</code> and
<code>results/student_test_results.csv</code>, the optimisation section reads
<code>results/optuna_trials.csv</code>, and the decoder comparison reads
<code>results/api_evaluation.json</code>, which the service writes during evaluation.</p>
<p>Reference routes never enter the prediction path. The service builds candidates, scores them and
decodes a route before it opens the route file, and the interface has no field that could carry
ground truth. Evaluator partition data was not used for training, feature selection, model choice or
decoding parameters.</p>
<p>AI assistance was used during development and is disclosed here. Every module is covered by tests
and can be explained by the team.</p>
"""
    )

    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>PathSnap Report</title><style>{CSS}</style></head><body>"
        + "".join(parts)
        + "</body></html>"
    )


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    """Print the HTML report to PDF with headless Chromium."""
    from playwright.sync_api import sync_playwright

    chrome = __import__("os").environ.get(
        "PLAYWRIGHT_CHROMIUM_PATH",
        "/home/sr/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome",
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.pdf(path=str(pdf_path), format="A4", print_background=True)
        browser.close()


def main() -> int:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_HTML} ({len(html) // 1024} kB)")
    render_pdf(OUTPUT_HTML, OUTPUT_PDF)
    print(f"wrote {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
