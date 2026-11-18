# Person 1 integration handoff

Read `docs/DATA_ENGINEERING.md` for definitions, assumptions and regeneration commands; `results/COMPLETION_REPORT.md` for measured results from this run.

## Anuvidya: classifier training

Read only `<8 digit trip ID>.parquet` in `data/processed/train/`. Do not glob alignment files into the model dataset. Keep only `training_eligible == True` (equivalently non-null labels). `X = rows[FEATURE_COLUMNS]`, `y = rows['label'].astype(int)`. Import `FEATURE_COLUMNS` from `src.data.features` and persist that ordered list with the trained model.

Split by trip with GroupKFold or GroupShuffleSplit before fitting imputation or scaling; never randomly split candidate rows from the same GPS observation. If geographic overlap between trips matters, add geographic holdouts because trip grouping alone cannot eliminate shared-road leakage. Preserve `(trip_id, gps_index, arc_id)` for joining model emissions back to candidates. Treat confidence as a diagnostic or optional sample weight, never an input feature. Missing true candidates have no supervised rows, and disconnected routes can have no eligible labels at all. Review per-trip coverage and exclusion rates before model evaluation. Student-test exports are forbidden as training or preprocessing inputs.

## Kruthika: FastAPI and HMM

```python
from src.data.pipeline import build_inference_features
from src.data.features import FEATURE_COLUMNS

rows = build_inference_features(dataset_folder, trip_id)
X = rows[FEATURE_COLUMNS]  # use the persisted training imputer/model
keys = rows[['trip_id', 'gps_index', 'arc_id']]
# model.predict_proba(X): convert probabilities to appropriately clipped log emissions
```

For uploads or existing in-memory tables, use `build_road_network`, `generate_candidates`, then `build_candidate_features` with original observation indexes. No reference route is needed. Group by GPS index and keep every directed candidate identity. For HMM transitions use road topology plus `position_m`, `length_m`, `from_node_id`, `to_node_id`; add remaining distance along the source candidate and offset into the destination. Movement between two positions on the same arc needs its own direction-aware case. Disconnected transitions should be impossible or explicitly penalised. Do not use Euclidean point distance as road-network traversal distance. `network.adjacency` uses row positions mapped through `node_ids`, and `network.arcs` preserves parallel edge IDs. `rows.attrs['crs']` and persisted metadata identify metre coordinates. Handle first/zero/negative-gap NaNs with the same trained preprocessing. Radius, candidate count and schema must match training unless validated as a model configuration change.

## Shakthi Raja: map display

Plot `projected_longitude` and `projected_latitude` in WGS84; these are nearest points on candidate roads. Join model/HMM selections using all three keys. Candidate rank is not arc ID and GPS index is not route occurrence. Fetch original arc endpoint coordinates through nodes/arcs or transform `start_x/start_y/end_x/end_y` with the trip CRS. Draw selected arcs or a viewport subset instead of the whole road network. Each trip has its own projection; do not combine projected x/y across trips without transforming to a common CRS.

## Shared contract

The candidate key, ordered 18-feature schema, directed endpoint IDs and WGS84 output coordinates are identical between training and inference. `results/output_schema.csv`, `candidate_coverage.csv`, `alignment_status_counts.csv` and `completion_metrics.json` document actual exported data. Excluded labels are null, not zero. No automatic commit, push or merge was performed.
