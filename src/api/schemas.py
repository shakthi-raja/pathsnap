"""Request and response models for the PathSnap API.

The payload shape is fixed: the interface was built against it before the
backend existed, so these models are the contract rather than an implementation
detail.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["PredictRequest", "PredictResponse", "TripList"]


class PredictRequest(BaseModel):
    """Body of ``POST /predict``: only a trip identifier crosses the wire."""

    trip_id: str = Field(min_length=1, max_length=32)


class TripList(BaseModel):
    """Response of ``GET /trips``."""

    trip_ids: list[str]


class PredictResponse(BaseModel):
    """Response of ``POST /predict``.

    Coordinates are ``[latitude, longitude]`` because that is the order Leaflet
    expects. Raw dataset files store longitude first; the swap happens in the
    service layer.
    """

    trip_id: str
    status: str
    track_points: list[list[float]]
    road_context: list[list[list[float]]]
    predicted_polylines: list[list[list[float]]]
    metrics: dict[str, float | int | None]
    runtime_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)
