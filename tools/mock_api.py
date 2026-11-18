"""Development fixture backend for the PathSnap UI.

This is **not** the production API. It exists so Person 4 can build and test
the interface before Kruthika's real backend lands, and it is marked as a
fixture in every response (``status: "fixture"``).

What it does
------------
Serves the agreed contract (``GET /health``, ``GET /trips``, ``POST /predict``)
using real trip geometry from the dataset, and produces ``predicted
polylines`` with naive nearest-arc snapping. There is no trained model and no
HMM, so the geometry is plausible but meaningless as a result.

What it deliberately does **not** do
------------------------------------
* It never reads ``.route``. The fixture therefore also proves that the UI can
  render a prediction without any reference data crossing the wire.
* It reports ``metrics`` as ``null``. A fixture must never look like evidence.

Usage::

    python tools/mock_api.py --data-root ../data/raw --port 8000
"""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("mock_api")

FIXTURE_WARNING = (
    "FIXTURE RESPONSE: naive nearest-arc snapping. No trained model, no HMM, "
    "no evaluation. Built for interface testing. Never cite as a result."
)

_MAX_ROAD_SEGMENTS = 4_000
_MAX_ROUTE_POINTS = 600
_CHUNK = 64


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    trip_id: str = Field(min_length=1)


def _load_arrays(trip_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a trip as ``(track, nodes, arcs)`` in lon/lat source order."""
    stem = trip_dir.name
    track = np.loadtxt(trip_dir / f"{stem}.track", ndmin=2)[:, :2]
    nodes = np.loadtxt(trip_dir / f"{stem}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{stem}.arcs", dtype=np.int64, ndmin=2)
    return track, nodes, arcs


def _to_display(points: np.ndarray) -> list[list[float]]:
    """Convert lon/lat arrays to ``[lat, lon]`` display order."""
    return [[float(lat), float(lon)] for lon, lat in points]


def _nearest_arc_indices(track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    """Index of the arc whose midpoint is closest to each GPS point.

    Deliberately crude: midpoints and straight-line distance. Uses the
    expanded squared-distance identity so memory stays bounded on trips whose
    network holds hundreds of thousands of arcs.
    """
    midpoints = (nodes[arcs[:, 0]] + nodes[arcs[:, 1]]) / 2.0
    mid_sq = (midpoints**2).sum(axis=1)
    result = np.empty(len(track), dtype=np.int64)
    for start in range(0, len(track), _CHUNK):
        chunk = track[start : start + _CHUNK]
        chunk_sq = (chunk**2).sum(axis=1)[:, None]
        distances = chunk_sq + mid_sq[None, :] - 2.0 * (chunk @ midpoints.T)
        result[start : start + _CHUNK] = distances.argmin(axis=1)
    return result


def _build_road_context(
    nodes: np.ndarray, arcs: np.ndarray, track: np.ndarray
) -> list[list[list[float]]]:
    """Road segments inside the trip's bounding box, capped for payload size."""
    pad = 0.01
    min_lon, min_lat = track.min(axis=0) - pad
    max_lon, max_lat = track.max(axis=0) + pad

    def inside(selection: np.ndarray) -> np.ndarray:
        # `selection` holds source-order (lon, lat) rows.
        return (
            (selection[:, 0] >= min_lon)
            & (selection[:, 0] <= max_lon)
            & (selection[:, 1] >= min_lat)
            & (selection[:, 1] <= max_lat)
        )

    keep = inside(nodes[arcs[:, 0]]) | inside(nodes[arcs[:, 1]])
    selected = arcs[keep]
    if len(selected) > _MAX_ROAD_SEGMENTS:
        stride = int(np.ceil(len(selected) / _MAX_ROAD_SEGMENTS))
        selected = selected[::stride]
    return [_to_display(nodes[arc]) for arc in selected]


def _build_predicted_polylines(
    track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray
) -> list[list[list[float]]]:
    """Return the snapped-midpoint path as a single polyline.

    A real decoder produces connected routes; this fixture only demonstrates
    that the UI can render whatever geometry the backend returns.
    """
    nearest = _nearest_arc_indices(track, nodes, arcs)
    midpoints = (nodes[arcs[nearest, 0]] + nodes[arcs[nearest, 1]]) / 2.0
    if len(midpoints) > _MAX_ROUTE_POINTS:
        stride = int(np.ceil(len(midpoints) / _MAX_ROUTE_POINTS))
        midpoints = midpoints[::stride]
    return [_to_display(midpoints)]


def create_app(data_root: Path) -> FastAPI:
    """Build the fixture application bound to a dataset root."""
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")

    trips: dict[str, Path] = {}
    for partition in ("train", "student_test"):
        partition_dir = data_root / partition
        if not partition_dir.is_dir():
            continue
        for trip_dir in sorted(partition_dir.iterdir()):
            if trip_dir.is_dir() and (trip_dir / f"{trip_dir.name}.track").is_file():
                trips[trip_dir.name] = trip_dir
    if not trips:
        raise SystemExit(f"No trips found under {data_root}")

    logger.info("Fixture serving %d trips from %s", len(trips), data_root)
    app = FastAPI(title="PathSnap fixture API", version="0.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "mode": "fixture", "trips": len(trips)}

    @app.get("/trips")
    def list_trips() -> dict[str, Any]:
        return {"trip_ids": sorted(trips)}

    @app.post("/predict")
    def predict(payload: PredictRequest) -> dict[str, Any]:
        trip_dir = trips.get(payload.trip_id)
        if trip_dir is None:
            raise HTTPException(status_code=404, detail=f"Unknown trip {payload.trip_id!r}")

        logger.info("POST /predict trip=%s", payload.trip_id)
        track, nodes, arcs = _load_arrays(trip_dir)
        return {
            "trip_id": payload.trip_id,
            "status": "fixture",
            "track_points": _to_display(track),
            "road_context": _build_road_context(nodes, arcs, track),
            "predicted_polylines": _build_predicted_polylines(track, nodes, arcs),
            "metrics": {"pointwise_accuracy": None, "route_continuity": None},
            "runtime_ms": None,
            "warnings": [FIXTURE_WARNING],
        }

    return app


def _lan_hint() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return "<this machine's LAN IP>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../data/raw"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.data_root.resolve())
    logger.info("UI should point PATHSNAP_API_URL at http://%s:%d", _lan_hint(), args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
