# PathSnap data engineering

Run commands from `D:\ML_HACKATHON\pathsnap` on `feature/data-engineering` with the existing Python 3.11 environment. No model fitting, commits, pushes, merges or raw-data edits are part of this pipeline.

## Layout and execution

- `data/raw/train/<trip_id>/`: 80 original training trips; zero-based row numbers are node, arc and observation IDs.
- `data/raw/student_test/<trip_id>/`: 10 held-out trips; inference only.
- `src/data/reader.py`: schema validation; default four-tuple remains compatible with the original reader.
- `exploration.py`, `network.py`, `candidates.py`, `features.py`, `labels.py`, `pipeline.py`, `report.py`: audit, projected geometry, indexed search, features, training-only alignment, export and verification.
- `tests/test_data_pipeline.py`: synthetic and real integration tests.
- `data/processed/train/`: per-trip candidate and alignment Parquet files, run metadata and processing summary.
- `data/processed/student_test_inference/`: separate unlabelled exports, metadata and summary.
- `results/`: exploration plots/tables, coverage, schema, test XML, completion metrics and source hashes.

PowerShell commands:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m src.data.exploration
.\.venv\Scripts\python.exe -m src.data.pipeline --trip 00000001 --output data/processed/single
.\.venv\Scripts\python.exe -m pytest -q --junitxml=results/test-results.xml
.\.venv\Scripts\python.exe -m src.data.pipeline --output data/processed/train
.\.venv\Scripts\python.exe -m src.data.pipeline --folder data/raw/student_test --output data/processed/student_test_inference --inference
.\.venv\Scripts\python.exe -m src.data.report
```

Candidate options: `--radius 50 --max-radius 200 --candidates 10`. `--trip` may be repeated. Use a fresh output directory for each configuration; outputs for selected IDs are replaced, unrelated prior outputs are retained. An export failure creates an error traceback and a failed summary row, and the CLI exits nonzero. A feature export can succeed even if every label is uncertain: inspect label counts separately. Large generated artifacts, results, raw data and virtual environments are Git-ignored.

## Python interfaces

```python
from src.data.reader import load_trip
from src.data.network import build_road_network
from src.data.candidates import generate_candidates, CandidateConfig
from src.data.features import build_candidate_features, FEATURE_COLUMNS
from src.data.labels import align_reference_route
from src.data.pipeline import build_training_dataset, build_inference_features

track, nodes, arcs, route = load_trip('data/raw/train', '00000001')
network = build_road_network(nodes, arcs)
candidates = generate_candidates(track, network, '00000001', CandidateConfig())
features = build_candidate_features(track, network, candidates)
alignment = align_reference_route(track, network, route)
training_rows, alignment = build_training_dataset('data/raw/train', '00000001')
inference_rows = build_inference_features('data/raw/student_test', '00000000')
```

The inference entry point calls `load_trip(..., include_route=False)` and never opens `.route`. Training entry points reject paths with a `student_test` directory component. No preprocessing is fitted during either pipeline. Invalid/nonfinite coordinates, malformed schemas and invalid references fail explicitly; observations are never silently renumbered or dropped by the file pipeline. Nonpositive time gaps remain represented with missing speed and a validity flag.

## Geometry and directed graph

Each arc is a straight segment from its original from-node to its to-node; intermediate road curvature is unavailable. IDs, duplicate connections, overlapping segments, reverse arcs and zero-length arcs survive processing. Collapsed segments use Point geometry with length zero and undefined orientation. A trip-local WGS84 azimuthal equidistant CRS is centred on the network (circular mean longitude handles the antimeridian). Projected geometry is in metres; networks beyond 500 km from the centre are rejected. This local approximation is appropriate for these networks, not a claim of distortion-free distances everywhere.

`network.arcs` includes endpoints, lengths and orientation. `network.adjacency` is a directed SciPy CSR matrix of minimum edge length, with parallel edges reduced by minimum rather than summation. Its positions map to original node IDs through `network.node_ids`. Arc identity remains in `network.arcs` for multiedge traversal and routing output. Explicit zero-weight edges must be retained when using SciPy routing routines; do not convert the graph to a dense matrix that treats zero as absent.

STRtree queries retrieve all roads within the search radius, then sort by true projected point-to-segment distance and arc ID, retaining at most ten. Radius expands only when the initial query is empty. If even the maximum radius is empty, a global nearest query returns tied geometries before deterministic truncation. `fallback_used` flags expansion/global fallback. Radius is a search setting, not a guarantee of final distance. Fewer than ten candidates is valid.

## Exact model schema

Use `FEATURE_COLUMNS` in the following order; do not infer model columns by selecting all numeric fields.

| Feature | Definition / units |
|---|---|
| distance_m | GPS-to-segment perpendicular distance, metres |
| candidate_rank | Distance rank, starting at one |
| position_fraction | Directed offset / segment length, zero for collapsed segment |
| position_m | Directed offset from from-node, metres |
| length_m | Segment length, metres |
| road_sin, road_cos | Sine/cosine of directed segment orientation |
| movement_m | Distance from previous GPS observation, metres |
| heading_sin, heading_cos | Sine/cosine of movement heading; missing for movement below 1 m |
| heading_difference_rad | Smallest angular difference in [0, pi], radians |
| dt_s | Timestamp difference from preceding observation, seconds |
| speed_mps | Movement / positive time gap, metres/second; otherwise missing |
| valid_time_gap | 1 for finite, positive gap; otherwise 0 |
| stationary | 1 for movement below 1 m; first observation is 0 with missing movement |
| zero_length | 1 for collapsed segment |
| gaussian_emission_20m | exp(-0.5 * (distance / 20)^2) |
| log_emission_20m | -0.5 * (distance / 20)^2 |

There are 18 model features. Angles are clockwise from north, modulo 2 pi. All movement features use only the previous observation, consistently in training and inference. The 20 m emission scale is a fixed baseline, not a fitted GPS-noise estimate or calibrated likelihood. First observations, stationary headings, zero-length orientations and invalid time gaps produce documented NaNs. Fit any imputer/scaler on training folds only. No feature uses labels, route membership or route progression. Arc IDs and absolute location are metadata, not transferable numeric predictors.

## Export schema

`results/output_schema.csv` records exact column names and pandas dtypes from the saved Parquet export. Train and inference share all columns below except the final label group:

- Keys: `trip_id` (zero-padded string), `gps_index` (original row ID), `arc_id` (original directed road ID); unique as a triple.
- Candidate metadata: `candidate_rank`, `distance_m`, `projected_x`, `projected_y` (trip CRS metres), `projected_longitude`, `projected_latitude` (WGS84 degrees), `position_m`, `position_fraction`, `search_radius_m`, `fallback_used`.
- GPS features/metadata: `movement_m`, `gps_heading_rad`, `dt_s`, `speed_mps`, `valid_time_gap`, `stationary`.
- Road features/metadata: `from_node_id`, `to_node_id`, `length_m`, `orientation_rad`, `start_x`, `start_y`, `end_x`, `end_y`, `road_sin`, `road_cos`, `heading_sin`, `heading_cos`, `heading_difference_rad`, `zero_length`, `log_emission_20m`, `gaussian_emission_20m`.
- Training only: `aligned_arc_id`, `route_occurrence`, `route_progress_m`, `alignment_distance_m`, `alignment_margin`, `alignment_confidence`, `alignment_confident`, `alignment_status`, `reference_in_candidates`, `training_eligible`, `label` (nullable Int8: 1 positive, 0 negative, null excluded).

The alignment Parquet contains one row per GPS index and the training alignment columns except `training_eligible` and `label`. Route occurrence is a zero-based index into `.route`, distinct from arc ID. Never pass it or any other training-only field to the model.

`metadata.json` stores schema version, exact ordered features, configurations, package/Python versions, input hashes and each trip CRS. `processing_summary.csv` stores success/error status, counts, candidate recall, elapsed seconds and process high-water memory in MiB. Peak memory is cumulative for the process, not per-trip allocation. `source_sha256.json` and `environment-lock.txt` capture the verified source and installed dependencies.

## Alignment objective and uncertainty

Reference route rows describe ordered directed arc occurrences, not GPS labels. Each GPS point is projected onto route segments found within 100 m; if none exist, nearest reference occurrences are used. Up to 48 nearest occurrence states survive; hitting the cap marks the observation uncertain. Each state retains its route occurrence and cumulative directed metre position, so repeated arcs and loops stay distinct.

Dynamic programming minimises squared distance / (2 * 20 m squared), plus absolute disagreement between route displacement and observed GPS displacement, scaled by max(20 m, half the GPS displacement). It adds penalties for backwards jitter and displacement faster than 60 m/s when a positive time gap exists. Occurrence indices cannot decrease; progress can retreat at most 5 m for noise. It permits many observations on one arc and arbitrarily many intervening route arcs between observations. No requirement forces a partial track to start/end on the route endpoints.

Forward/backward min-sum scores provide a cost margin against other occurrence states. A single coherent path is recovered for each block. If no monotonic path exists, alignment restarts explicitly and adjacent observations are excluded. Distance above 35 m, margin below 2 cost units, a state cap, invalid/over-30-second gap, or disconnected reference also excludes labels. Any connectivity break conservatively excludes the whole trip's labels. Confidence combines a distance kernel and margin saturation; it is a heuristic, not ground-truth certainty or a calibrated probability.

An observation is training eligible only if alignment is confident AND its aligned arc appears among generated candidates. All candidate labels for ineligible observations are null, avoiding false all-negative examples. Candidate recall is covered confident observations / all confident observations, including confident observations missing their arc. Coverage does not validate alignment correctness and says nothing about uncertain observations. Inspect per-trip recall; a high pooled number can hide a weak trip.

## Known limits and official checks

Alignment is conservative pseudo-supervision from route geometry, not manually verified GPS-to-arc truth. Closely spaced parallel roads, intersections, exact shared endpoints and loops can remain ambiguous. Truncated/search-limited state sets can miss a plausible alternative. Timestamp units are assumed seconds based on supplied specification and observed files. Candidate caps, straight-line geometry and long GPS gaps constrain recall. There is no trained classifier or full HMM decoder in Person 1 scope.

No official hackathon instructions were present locally. Verify permitted reference-route use, timestamp semantics, official metrics, split rules, data licence and submission format with organisers. Held-out reference files were not read or evaluated. The pipeline provides integration contracts and measured diagnostics; it does not establish official compliance or final map-matching accuracy.
