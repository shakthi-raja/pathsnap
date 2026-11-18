"""Load precomputed candidate probabilities and recover their routing geometry."""

from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pandas.api.types import is_float_dtype, is_integer_dtype

from .network import RoadNetwork
from .reader import normalize_trip_id

KEYS = ["trip_id", "gps_index", "arc_id"]
SCORE_COLUMNS = [*KEYS, "p_correct"]


class EvaluatorInputError(ValueError):
    """Evaluator scores cannot be matched to the agreed input contract."""


def load_evaluator_scores(directory: Path) -> dict[str, pd.DataFrame]:
    """Read Parquet shards, independent of filenames, and group by trip."""
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No evaluator prediction Parquets in {directory}")

    tables = []
    for path in paths:
        try:
            scores = pd.read_parquet(path, columns=SCORE_COLUMNS)
            scores["trip_id"] = scores["trip_id"].map(normalize_trip_id)
            for key in ("gps_index", "arc_id"):
                ids = scores[key]
                if not is_integer_dtype(ids) or ids.isna().any() or (ids < 0).any():
                    raise ValueError(f"{key} must contain nonnegative integer IDs")

            probabilities = scores["p_correct"]
            if not (is_float_dtype(probabilities) or is_integer_dtype(probabilities)):
                raise ValueError("p_correct must be numeric")
            values = probabilities.to_numpy(dtype=float, na_value=np.nan)
            if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
                raise ValueError("p_correct must contain finite probabilities in [0, 1]")
            scores["p_correct"] = values
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise EvaluatorInputError(f"Invalid evaluator prediction file {path.name}: {exc}") from exc
        tables.append(scores)

    scores = pd.concat(tables, ignore_index=True)
    if scores.empty:
        raise EvaluatorInputError("Evaluator prediction Parquets contain no scored candidates")
    if scores.duplicated(KEYS).any():
        raise EvaluatorInputError("Duplicate evaluator candidate keys (trip_id, gps_index, arc_id)")
    scores = scores.sort_values(KEYS)
    return {trip_id: rows for trip_id, rows in scores.groupby("trip_id", sort=False)}


def evaluator_candidates(
    scores: pd.DataFrame, track: pd.DataFrame, network: RoadNetwork
) -> pd.DataFrame:
    """Project each scored observation onto its supplied arc in the raw graph."""
    observation_ids = scores["gps_index"].drop_duplicates()
    observation_positions = track.index.get_indexer(observation_ids)
    arc_positions = pd.Index(network.arcs["arc_id"]).get_indexer(scores["arc_id"])
    trip_id = scores["trip_id"].iloc[0]
    if (observation_positions < 0).any():
        raise EvaluatorInputError(f"Trip {trip_id}: evaluator gps_index is absent from the raw track")
    if (arc_positions < 0).any():
        raise EvaluatorInputError(f"Trip {trip_id}: evaluator arc_id is absent from the raw graph")

    observations = track.iloc[observation_positions]
    xy = network.project(observations)
    if not np.isfinite(xy).all():
        raise EvaluatorInputError(f"Trip {trip_id}: evaluator observations could not be projected")
    # Measure from the previous scored observation, including any gap in coverage.
    delta = np.diff(xy, axis=0, prepend=xy[:1])
    groups = pd.Index(observation_ids).get_indexer(scores["gps_index"])
    points = shapely.points(xy[groups])
    arcs = network.arcs.iloc[arc_positions]
    geometries = network.geometries[arc_positions]
    nonzero = arcs["length_m"].to_numpy() > 0
    lengths = np.zeros(len(scores), dtype=float)
    lengths[nonzero] = shapely.line_locate_point(geometries[nonzero], points[nonzero])
    snapped = geometries.copy()
    snapped[nonzero] = shapely.line_interpolate_point(geometries[nonzero], lengths[nonzero])

    result = scores.copy()
    for column in ("from_node_id", "to_node_id", "length_m"):
        result[column] = arcs[column].to_numpy()
    result["position_m"] = lengths
    result["projected_x"] = shapely.get_x(snapped)
    result["projected_y"] = shapely.get_y(snapped)
    result["movement_m"] = np.hypot(delta[:, 0], delta[:, 1])[groups]
    result["dt_s"] = observations["timestamp"].diff().to_numpy()[groups]
    return result
