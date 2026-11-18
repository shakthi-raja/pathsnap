# Data engineering completion report

Verified workspace: `D:\ML_HACKATHON\pathsnap`. Branch: `feature/data-engineering`. No commit, push or merge was performed. Raw inputs were read only; student-test references were not read. Installed and used pytest in the existing Python 3.11 virtual environment.

## Executed results

- Automated suite: **39 passed**, 0 failed; `results/test-results.xml`.
- Training export: **80/80 trips**, 0 failures; **208,417 GPS observations**, **1,304,458 candidate rows**.
- Label alignment: 90,043 confident, **118,374 uncertain** observations.
- Ambiguous-occurrence flag: 62,204 observations; disconnected-reference flag: 35,775; alignment-reset flag: 3,178. Flags can overlap and are not additive.
- Supervised labels: **89,001 positive**, **391,257 negative**, **824,200 null/excluded candidate rows**.
- Confident observations missing their aligned arc: 1,042; all their candidate labels are null.
- Expanded/global search: 18,230 observations.
- Held-out inference export: **10/10 trips**, **19,160 observations**, **131,528 candidates**, no reference inputs or labels.
- Persisted-file integration contracts verified across 80 training and 10 inference trips.
- Sum of training trip processing times: 434.667 seconds; process high-water memory: **301.30 MiB**, including native array/geometry allocations. This is one measured run, not a cross-platform benchmark.

## Candidate coverage

Weighted coverage over confident aligned observations only; uncertain observations are excluded from the denominator. This measures retrieval, not ground-truth alignment accuracy.

- Top 1: 49.5808% (44,644/90,043).
- Top 5: 93.1244% (83,852/90,043).
- Top 8: 97.9610% (88,207/90,043).
- Top 10: 98.8428% (89,001/90,043).

Ten candidates improve retrieval but cannot guarantee coverage in dense/parallel/duplicate roads. Lowest nonempty per-trip coverage:

- 00000072: 72.47%.
- 00000016: 81.62%.
- 00000061: 86.86%.
- 00000010: 87.01%.
- 00000054: 92.57%.

## Quality findings and implications

The 80-trip audit found 0 invalid GPS coordinates, 0 invalid node coordinates, 0 invalid node references, 0 invalid route references and 0 exact duplicate GPS rows. Missing GPS/node values total 0.

There are 572 repeated GPS coordinate pairs, 1775 duplicate directed road connections, 22 self-loop arcs, 133 time gaps over 30 seconds, and 553 approximate GPS speeds over 60 m/s. There are 0 negative gaps and 0 repeated timestamps. Maximum timestamp gap: 153,064 seconds. Weak component counts range from 1 to 421; use directed graph reachability in HMM transitions instead of assuming all nearby roads connect.

Reference connectivity breaks total 13 across 10 trips. 10 trips have no confident supervision: 00000002, 00000019, 00000033, 00000041, 00000042, 00000069, 00000075, 00000080, 00000086, 00000089. Features remain available; do not train on their null labels. Trip 00000054 is a major fallback/outlier case (see per-trip summary). Long gaps should trigger cautious HMM handling; repeated locations can have undefined headings. Duplicates/reverse arcs remain distinct candidates.

## Files and outputs

Modified: `.gitignore`, `src/data/reader.py`, `exploration.py`, `candidates.py`, `features.py`, `labels.py`. Created: `src/data/network.py`, `pipeline.py`, `report.py`, `tests/test_data_pipeline.py`, `pytest.ini`, `requirements-data.txt`, `docs/DATA_ENGINEERING.md`, `docs/INTEGRATION_HANDOFF.md`, this report. Existing data-package init was retained. All work remains uncommitted.

Primary outputs:

- `data/processed/train/<trip_id>.parquet`: candidate metadata, 18 model features, nullable labels and alignment diagnostics.
- `data/processed/train/<trip_id>.alignment.parquet`: one alignment row per GPS observation.
- `data/processed/student_test_inference/<trip_id>.parquet`: same candidate features without route/label fields.
- Both output directories: `metadata.json` (versions, configs, input SHA256, per-trip CRS), `processing_summary.csv` (all statuses/counts/timings/memory).
- `results/exploration/trip_summary.csv` and `.parquet`, four trajectory plots, and `distributions.png`.
- `results/candidate_coverage.csv`, `alignment_status_counts.csv`, `output_schema.csv`, `unusual_trips.csv`, `quality_totals.json`, `completion_metrics.json`, `test-results.xml`, `source_sha256.json`, `environment-lock.txt`.

Raw data, virtual environments, results and processed Parquet remain Git-ignored. Final datasets are in `train/`; `smoke/` contains development runs and is not the final handoff.

## Model feature schema

Ordered model columns: `distance_m`, `candidate_rank`, `position_fraction`, `position_m`, `length_m`, `road_sin`, `road_cos`, `movement_m`, `heading_sin`, `heading_cos`, `heading_difference_rad`, `dt_s`, `speed_mps`, `valid_time_gap`, `stationary`, `zero_length`, `gaussian_emission_20m`, `log_emission_20m`.

Feature definitions and units are in `docs/DATA_ENGINEERING.md`; exact saved column names/dtypes are in `results/output_schema.csv`. Arc IDs, observation IDs, trip IDs, absolute coordinates, route progress and labels are excluded from model predictors. Headings are clockwise from north; distance is metres, time seconds and speed metres/second. First-observation/invalid-time/stationary/zero-length cases retain documented NaNs, to be imputed using training-only fitted preprocessing.

## Teammate integration

Anuvidya: read only 8-digit training candidate files, filter `training_eligible`, select the exact ordered `FEATURE_COLUMNS`, split by trip before fitting preprocessing/model, and keep the three-key candidate identity. Do not treat null labels as negative or use student-test data in fitting.

Kruthika: call `build_inference_features` or the shared network/candidate/feature functions; apply the training model and preprocessing to the same columns. Group emissions by GPS index and use directed graph distances plus candidate offsets for HMM transitions. No route files are required. Road node row positions map through `network.node_ids`.

Shakthi Raja: use `projected_longitude`/`projected_latitude` for map points and join selected candidates by `(trip_id, gps_index, arc_id)`. Segment endpoints are also available in each trip's metric CRS. Do not mix projected coordinates between trips without a common transform. See `docs/INTEGRATION_HANDOFF.md`.

## Actual principal commands executed

```powershell
Get-Location
git branch --show-current
git status --short
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m src.data.exploration
.\.venv\Scripts\python.exe -m src.data.pipeline --trip 00000001 --output data/processed/smoke
.\.venv\Scripts\python.exe -m src.data.pipeline --trip 00000001 --trip 00000002 --trip 00000003 --output data/processed/smoke
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q --junitxml=results/test-results.xml
.\.venv\Scripts\python.exe -u -m src.data.pipeline --output data/processed/train
.\.venv\Scripts\python.exe -u -m src.data.pipeline --folder data/raw/student_test --output data/processed/student_test_inference --inference
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 results/environment-lock.txt
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m src.data.report
```

Additional commands inspected raw samples, versions, statistics and Git ignore rules. A development smoke run exposed an arc-index/column pandas merge ambiguity, which was fixed. The first test run was 27 passed/1 failed: a loop fixture repeated stationary coordinates at a turn and incorrectly required a unique occurrence; the corrected fixture traverses both turn endpoints and checks known interior occurrences. Subsequent suites passed 34, 35 and finally 39 tests. Pip initially failed under restricted DNS; the approved retry installed pytest. The matplotlib cache now stays under project results.

## Readiness and remaining limits

The data-engineering interfaces and exports are ready for teammate integration, with verified schemas and explicit exclusion policies. They do not establish final model/HMM accuracy. Labels are conservative heuristic sequence alignment, not manually verified point truth. Shared endpoints, overlapping route occurrences, state/search truncation and long gaps can remain ambiguous. Disconnected routes conservatively lose all supervision. Candidate retrieval misses some confident reference arcs; inspect per-trip recall. Road shape is approximated by supplied node-to-node segments; the local projection rejects networks extending over 500 km from its centre. Confidence is not calibrated. A trained classifier and HMM decoder are outside this assignment.

Official hackathon instructions were not present locally. Permitted reference use, timestamp semantics, official metrics, split rules, licensing and submission format remain unverified and require organiser confirmation. Student-test references were neither evaluated nor used for fitting. No claim of official compliance is made.
