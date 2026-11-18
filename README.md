# PathSnap

Distributed map-matching and route reconstruction for noisy GPS traces.
UE24CS352A Machine Learning hackathon, PES University.

A vehicle's GPS drifts in cities, so a car on one street can look like it is on
the parallel one. PathSnap matches each GPS observation to a road segment and
reconstructs a route that a vehicle could actually have driven.

## How it works

```
track + nodes + arcs
        |
        v
candidate generation        every road segment within 50 m of an observation
        |
        v
feature extraction          18 scale-free features per (observation, candidate)
        |
        v
candidate classifier        predicts whether a candidate is the reference match
        |
        v
Viterbi decoder             picks a connected path using the road graph and time
        |
        v
predicted route + score     scored server-side against the reference route
```

Two design decisions drive everything else:

1. **The model scores candidates, it does not predict arc IDs.** Arc IDs are row
   numbers inside one trip, so they mean nothing across the 77 cities in the
   dataset. Each observation has several plausible roads; the model ranks them.
2. **Reference labels never reach the predictor.** The route file is opened only
   after a prediction exists, inside the server, to compute the score.

## Repository layout

| Path | Contents |
|---|---|
| `src/data/` | dataset reading, road network, candidate generation, features, reference alignment |
| `src/ml/` | dataset assembly, model training, evaluation metrics, tuning, export |
| `src/routing/` | Viterbi decoder and transition scoring |
| `src/api/` | FastAPI backend, prediction service, server-side evaluation |
| `src/ui/` | evaluator interface and typed HTTP client |
| `src/visualization/` | Folium map builder |
| `tools/` | fixture backend for UI development, report map generator |
| `tests/` | unit tests |
| `results/` | experiment outputs, comparison tables, Optuna trials |
| `artifacts/` | trained pipeline and metadata |

## Requirements

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements-data.txt` lists the exact versions used to build the processed
tables.

## Data

The raw dataset is **not** in this repository. It contains 90 trips: 80 for
training and 10 for student testing, each with four files (`.track`, `.nodes`,
`.arcs`, `.route`). The evaluator partition is supplied separately on the day.

Point the code at it with an environment variable:

```bash
export PATHSNAP_DATA_ROOT=/path/to/data/raw       # fish: set -x PATHSNAP_DATA_ROOT ...
```

The directory must contain `train/` and `student_test/`.

## Building the training tables

```bash
# 80 training trips, produces one parquet per trip plus an alignment sidecar
python -m src.data.pipeline \
    --folder "$PATHSNAP_DATA_ROOT/train" \
    --output "$PATHSNAP_DATA_ROOT/../processed/train"

# ten student-testing trips, labelled for development evaluation only
python -m src.data.pipeline \
    --folder "$PATHSNAP_DATA_ROOT/student_test" \
    --output "$PATHSNAP_DATA_ROOT/../processed/student_test" \
    --evaluation-export
```

`--evaluation-export` is required for the student-test partition. Without it the
pipeline refuses, because labelled exports are meant for evaluation and never for
fitting.

## Training and comparison

```bash
python -m src.ml.train_models      # five models, grouped CV, student-test scores
python -m src.ml.tune_model        # Optuna TPE search for the chosen model
```

Outputs:

| File | Contents |
|---|---|
| `results/cv_fold_results.csv` | every fold, every model |
| `results/model_comparison.csv` | CV mean and standard deviation joined with student-test scores |
| `results/student_test_results.csv` | per-model student-test metrics and timing |
| `results/optuna_trials.csv` | full Bayesian search record |
| `artifacts/final_pipeline.joblib` | selected model |
| `artifacts/model_metadata.json` | feature order and selection evidence |

## Running the system

**Laptop 1, the backend:**

```bash
export PATHSNAP_EVALUATION_ROOT=/path/to/evaluation/trips
python -m src.api.main                 # serves on 0.0.0.0:8000
```

**Laptop 2, the interface:**

```bash
export PATHSNAP_API_URL=http://<laptop-1-lan-ip>:8000
python -m streamlit run src/ui/app.py
```

`127.0.0.1` on Laptop 2 means Laptop 2, not Laptop 1. Use the LAN address.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, trip count, loaded model |
| `GET /trips` | trip IDs discovered on the server, never hardcoded |
| `POST /predict` | full pipeline for one trip |

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/trips
curl -s -X POST http://127.0.0.1:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"trip_id":"00000009"}'
```

Response, abbreviated:

```json
{
  "trip_id": "00000009",
  "status": "ok",
  "track_points": [[41.86, 12.56], "..."],
  "road_context": [[[41.86, 12.56], [41.86, 12.57]], "..."],
  "predicted_polylines": [[[41.86, 12.56], "..."], "..."],
  "metrics": {
    "pointwise_accuracy": 0.949,
    "top1_accuracy": 0.889,
    "route_continuity": 1.0,
    "classifier_continuity": 0.996,
    "candidate_miss_rate": 0.014,
    "evaluated_observations": 217
  },
  "runtime_ms": 3235,
  "warnings": []
}
```

Coordinates are `[latitude, longitude]` for Leaflet. Raw files store longitude
first; the swap happens once, in the service.

`metrics` is empty when the server holds no reference route for the trip.

## Development fixture

The interface can be developed without a trained model:

```bash
python tools/mock_api.py --data-root "$PATHSNAP_DATA_ROOT" --port 8000
```

It serves the same contract from real geometry, never reads `.route`, and marks
every response `status: "fixture"`. Not evidence, and not a demo.

## Report maps

```bash
python tools/render_trip_map.py \
    --trip 00000062 --partition train --data-root "$PATHSNAP_DATA_ROOT" \
    --out results/maps/trip_00000062_overview.html --with-reference
```

Produces the Folium evidence the report needs, including the reference route.
Only valid on training and student-test trips.

## Report

```bash
python tools/make_report_figures.py   # charts from results/*.csv
python tools/build_report.py          # renders report/PathSnap_Report.pdf
```

The builder reads the recorded CSVs and artifact metadata, so a regenerated
report always matches the run in this repository. No metric in the PDF is typed
by hand.

## Tests

```bash
python -m pytest
```

## Results

Five model families, grouped cross-validation over trips, scored on the ten
student-testing trips. See `results/model_comparison.csv` for the recorded run.
Selection uses observation-level Top-1, because most candidate rows are negatives
and row accuracy would reward a model that rejects everything.

## Academic integrity

Every result in the report traces to a file in `results/` or `artifacts/`
produced by the commands above. No metric is estimated or transcribed by hand.
Reference routes stay on the server; the interface has no code path that could
receive them.

## Team

| Person | Area |
|---|---|
| Person 1 | data engineering: reading, candidates, features, label alignment |
| Person 2 | model training, comparison, tuning |
| Person 3 | HMM decoding and backend |
| Person 4 | interface, maps, integration, report |
