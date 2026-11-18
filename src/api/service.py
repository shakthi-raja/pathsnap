"""Prediction service: the whole pipeline behind one request.

Order matters here and is enforced by construction. The service builds features
from the track and the road network, scores candidates, decodes a route, and
only then reaches for the reference route to score the result. The prediction
functions never receive the reference data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from shapely.ops import substring

from src.data.candidates import CandidateConfig, generate_candidates
from src.data.features import build_candidate_features
from src.data.labels import align_reference_route
from src.data.network import build_road_network
from src.data.reader import load_trip, normalize_trip_id
from src.routing.decoder import decode_candidates
from src.routing.reconstruction import reconstruct_route
from src.routing.transitions import NetworkTravel

from .evaluation import evaluate_prediction

logger = logging.getLogger(__name__)

__all__ = ["PathSnapService", "PredictionResult"]

#: Cap on road segments sent to the interface.
MAX_ROAD_SEGMENTS = 2500
#: Longest road link bridged inside one predicted polyline.
MAX_POLYLINE_JOIN_M = 120.0


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Everything the interface needs for one trip."""

    trip_id: str
    status: str
    track_points: list[list[float]]
    road_context: list[list[list[float]]]
    predicted_polylines: list[list[list[float]]]
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    runtime_ms: float | None = None
    warnings: list[str] = field(default_factory=list)


class PathSnapService:
    """Loads the trained pipeline once and serves predictions for any trip.

    Args:
        evaluation_root: Directory holding trip folders for evaluation or
            development. Each trip folder contains ``.track``, ``.nodes``,
            ``.arcs`` and, when scoring is possible, ``.route``.
        model_path: Saved scikit-learn pipeline.
        metadata_path: Feature order and model description written at export.
    """

    def __init__(
        self,
        evaluation_root: str | Path,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        candidate_config: CandidateConfig | None = None,
    ) -> None:
        self.evaluation_root = Path(evaluation_root)
        self.model = joblib.load(model_path)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.features: list[str] = self.metadata["features"]
        self.candidate_config = candidate_config or CandidateConfig()
        # Heavy objects are cached in a plain dict, never in DataFrame.attrs:
        # pandas deep-copies attrs on .copy(), and deep-copying a road network
        # with its spatial index is pathologically slow.
        self._cache: dict[str, tuple[pd.DataFrame, object, object]] = {}

        if not self.evaluation_root.is_dir():
            raise FileNotFoundError(f"Evaluation directory not found: {self.evaluation_root}")

    # -- discovery ---------------------------------------------------------
    def list_trips(self) -> list[str]:
        """Trip IDs available on the server, discovered from the filesystem."""
        trip_ids = sorted(
            path.name
            for path in self.evaluation_root.iterdir()
            if path.is_dir() and (path / f"{path.name}.track").is_file()
        )
        return trip_ids

    # -- pipeline ----------------------------------------------------------
    def _features_for(self, trip_id: str) -> tuple[pd.DataFrame, object, object]:
        """Candidate features, network and track for a trip.

        Building candidates is the expensive part, so the result is cached for
        the lifetime of the service.
        """
        cached = self._cache.get(trip_id)
        if cached is not None:
            return cached

        trip_dir = self.evaluation_root / trip_id
        track, nodes, arcs, _ = load_trip(trip_dir.parent, trip_id, include_route=False)
        network = build_road_network(nodes, arcs)
        candidates = generate_candidates(track, network, trip_id, self.candidate_config)
        frame = build_candidate_features(track, network, candidates)
        entry = (frame, network, track)
        self._cache[trip_id] = entry
        return entry

    def predict(self, trip_id: str) -> PredictionResult:
        """Run the full pipeline for one trip and score it if reference exists."""
        trip_id = normalize_trip_id(trip_id)
        started = time.perf_counter()
        warnings: list[str] = []

        frame, network, track = self._features_for(trip_id)

        features = frame[self.features].replace([np.inf, -np.inf], np.nan)
        probabilities = self.model.predict_proba(features)
        classes = list(self.model.classes_)
        positive = probabilities[:, classes.index(1)]

        travel = NetworkTravel(network.adjacency)
        route = decode_candidates(frame, positive, travel=travel)

        metrics: dict[str, float | int | None] = {}
        route_path = self.evaluation_root / trip_id / f"{trip_id}.route"
        if route_path.is_file():
            _, _, arcs, reference = load_trip(self.evaluation_root, trip_id)
            alignment = align_reference_route(track, network, reference)
            classifier_route = route.decoded[["gps_index"]].merge(
                route.classifier_only, on="gps_index", how="left"
            )
            classifier_route = classifier_route.merge(
                frame[["gps_index", "arc_id", "from_node_id", "to_node_id"]],
                on=["gps_index", "arc_id"],
                how="left",
            )
            metrics = evaluate_prediction(
                route.decoded,
                route.classifier_only,
                alignment,
                frame,
                travel,
                classifier_route=classifier_route,
            )
        else:
            warnings.append(
                "No reference route on the server for this trip, so accuracy is not reported."
            )

        if len(route.decoded) < len(track):
            warnings.append(
                f"{len(track) - len(route.decoded)} observations had no candidate within the search radius."
            )

        result = PredictionResult(
            trip_id=trip_id,
            status="ok",
            track_points=_track_points(track, network),
            road_context=self._road_context(network, track),
            predicted_polylines=_polylines(route.decoded, network, travel),
            metrics=metrics,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            warnings=warnings,
        )
        logger.info(
            "predict trip=%s observations=%d runtime_ms=%.0f accuracy=%s",
            trip_id,
            len(route.decoded),
            result.runtime_ms,
            metrics.get("pointwise_accuracy"),
        )
        return result

    # -- geometry ----------------------------------------------------------
    def _road_context(self, network, track) -> list[list[list[float]]]:
        """Road segments near the trajectory, in [lat, lon] order."""
        lon_min, lon_max = track.longitude.min(), track.longitude.max()
        lat_min, lat_max = track.latitude.min(), track.latitude.max()
        pad = 0.005
        segments: list[list[list[float]]] = []
        for geometry in network.geometries:
            coords = list(geometry.coords)
            if not coords:
                continue
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:], strict=True):
                lon1, lat1 = network.inverse.transform(x1, y1)
                if not (
                    lon_min - pad <= lon1 <= lon_max + pad
                    and lat_min - pad <= lat1 <= lat_max + pad
                ):
                    continue
                lon2, lat2 = network.inverse.transform(x2, y2)
                segments.append([[float(lat1), float(lon1)], [float(lat2), float(lon2)]])
        if len(segments) > MAX_ROAD_SEGMENTS:
            stride = int(np.ceil(len(segments) / MAX_ROAD_SEGMENTS))
            segments = segments[::stride]
        return segments


def _track_points(track, network) -> list[list[float]]:
    return [
        [float(lat), float(lon)] for lon, lat in zip(track.longitude, track.latitude, strict=True)
    ]


def _polylines(decoded: pd.DataFrame, network, travel) -> list[list[list[float]]]:
    """Render directed route pieces in the API's [latitude, longitude] order."""
    pieces = reconstruct_route(decoded, network, travel, max_join_m=MAX_POLYLINE_JOIN_M)
    positions = {arc_id: i for i, arc_id in enumerate(network.arcs["arc_id"])}
    polylines = []
    for piece in pieces:
        coordinates = []
        for part in piece:
            geometry = network.geometries[positions[part.arc_id]]
            clipped = substring(geometry, part.start_m, part.end_m)
            for point in clipped.coords:
                if not coordinates or coordinates[-1] != point:
                    coordinates.append(point)

        line = []
        for x, y in coordinates:
            lon, lat = network.inverse.transform(x, y)
            line.append([float(lat), float(lon)])
        if len(line) >= 2:
            polylines.append(line)
    return polylines
