"""Typed HTTP client for the PathSnap backend.

The client is the only place in the UI that knows about HTTP, JSON or the wire
format. Everything above it deals in :class:`Prediction` objects, so the
Streamlit layer stays thin and testable.

Design notes
------------
* **Dependency injection.** A ``requests.Session`` may be supplied, which is
  how the unit tests avoid touching the network.
* **Fail loudly, but usefully.** Transport failures, HTTP errors and malformed
  bodies raise distinct exceptions with messages an operator can act on.
* **Never trust the payload.** Field types are validated on arrival; a schema
  drift on the backend surfaces here rather than as a blank map.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests

from ..config import Settings

__all__ = [
    "ApiError",
    "ApiResponseError",
    "ApiUnavailable",
    "PathSnapClient",
    "Prediction",
]

logger = logging.getLogger(__name__)

_JSON_HEADERS = {"Accept": "application/json"}


class ApiError(RuntimeError):
    """Base class for failures talking to the PathSnap backend."""


class ApiUnavailable(ApiError):
    """The backend could not be reached (wrong host, firewall, server down)."""


class ApiResponseError(ApiError):
    """The backend responded, but the payload was unusable."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """A validated ``POST /predict`` response.

    Attributes:
        trip_id: Trip the prediction belongs to.
        status: Backend status string (``"ok"`` on success).
        track_points: Raw GPS observations as ``(lat, lon)``.
        road_context: Road segments as lists of ``(lat, lon)``.
        predicted_polylines: Reconstructed route pieces; several pieces mean
            the trajectory contained gaps.
        metrics: Backend-computed metrics. Values may be ``None`` when the
            backend does not provide them; the UI must show them as N/A rather
            than inventing a number.
        runtime_ms: Server-side runtime in milliseconds, if reported.
        warnings: Human-readable warnings raised during inference.
    """

    trip_id: str
    status: str
    track_points: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    road_context: tuple[tuple[tuple[float, float], ...], ...] = field(default_factory=tuple)
    predicted_polylines: tuple[tuple[tuple[float, float], ...], ...] = field(default_factory=tuple)
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    runtime_ms: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _require(payload: Mapping[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    if key not in payload:
        raise ApiResponseError(f"Response is missing required key {key!r}.")
    value = payload[key]
    if not isinstance(value, expected):
        names = (
            ", ".join(t.__name__ for t in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ApiResponseError(
            f"Response key {key!r} should be {names}, got {type(value).__name__}."
        )
    return value


def _parse_point(value: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ApiResponseError(f"{field_name} must contain [lat, lon] pairs, got {value!r}.")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ApiResponseError(f"{field_name} contains non-numeric values: {value!r}.") from exc


def _parse_line(value: Any, *, field_name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ApiResponseError(f"{field_name} must be a list of points.")
    return tuple(_parse_point(point, field_name=field_name) for point in value)


def _parse_lines(value: Any, *, field_name: str) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ApiResponseError(f"{field_name} must be a list of polylines.")
    return tuple(_parse_line(line, field_name=field_name) for line in value)


class PathSnapClient:
    """Thin, typed wrapper over the PathSnap HTTP API.

    Args:
        settings: Resolved application settings (base URL and timeouts).
        session: Optional pre-configured ``requests.Session``; one is created
            and owned by the client when omitted.
    """

    def __init__(self, settings: Settings, *, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._session = session or requests.Session()
        self._owns_session = session is None

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Release the underlying connection pool when this client owns it."""
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> PathSnapClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- endpoints ---------------------------------------------------------
    def health(self) -> Mapping[str, Any]:
        """Probe ``GET /health``.

        Raises:
            ApiUnavailable: The backend could not be reached.
            ApiResponseError: The response was not valid JSON.
        """
        url = f"{self._settings.api_url}/health"
        try:
            response = self._session.get(
                url, headers=_JSON_HEADERS, timeout=self._settings.health_timeout_s
            )
        except requests.RequestException as exc:
            raise ApiUnavailable(self._describe_transport_failure(url, exc)) from exc
        return self._decode(response, url)

    def list_trips(self) -> tuple[str, ...]:
        """Return the trip IDs the backend is currently able to serve.

        The list is always discovered from the server. The UI never hardcodes
        trip IDs, because the evaluator partition is supplied separately.

        Raises:
            ApiUnavailable: The backend could not be reached.
            ApiResponseError: The payload was malformed.
        """
        url = f"{self._settings.api_url}/trips"
        try:
            response = self._session.get(
                url, headers=_JSON_HEADERS, timeout=self._settings.api_timeout_s
            )
        except requests.RequestException as exc:
            raise ApiUnavailable(self._describe_transport_failure(url, exc)) from exc
        payload = self._decode(response, url)
        trip_ids = _require(payload, "trip_ids", (list, tuple))
        for trip_id in trip_ids:
            if not isinstance(trip_id, str):
                raise ApiResponseError(f"trip_ids must contain strings, got {trip_id!r}.")
        return tuple(trip_ids)

    def predict(self, trip_id: str) -> Prediction:
        """Run the full pipeline for one trip via ``POST /predict``.

        Args:
            trip_id: Identifier returned by :meth:`list_trips`.

        Raises:
            ValueError: ``trip_id`` is empty.
            ApiUnavailable: The backend could not be reached.
            ApiResponseError: The payload was malformed.
        """
        if not trip_id or not trip_id.strip():
            raise ValueError("trip_id must be a non-empty string.")

        url = f"{self._settings.api_url}/predict"
        try:
            response = self._session.post(
                url,
                json={"trip_id": trip_id},
                headers=_JSON_HEADERS,
                timeout=self._settings.api_timeout_s,
            )
        except requests.RequestException as exc:
            raise ApiUnavailable(self._describe_transport_failure(url, exc)) from exc
        payload = self._decode(response, url)

        trip_id_value = _require(payload, "trip_id", str)
        status = _require(payload, "status", str)
        metrics_raw = payload.get("metrics") or {}
        if not isinstance(metrics_raw, Mapping):
            raise ApiResponseError("metrics must be an object.")

        runtime = payload.get("runtime_ms")
        warnings_raw = payload.get("warnings") or []
        if not isinstance(warnings_raw, Sequence) or isinstance(warnings_raw, (str, bytes)):
            raise ApiResponseError("warnings must be a list of strings.")

        return Prediction(
            trip_id=trip_id_value,
            status=status,
            track_points=_parse_line(payload.get("track_points", []), field_name="track_points"),
            road_context=_parse_lines(payload.get("road_context", []), field_name="road_context"),
            predicted_polylines=_parse_lines(
                payload.get("predicted_polylines", []), field_name="predicted_polylines"
            ),
            metrics={str(k): (None if v is None else float(v)) for k, v in metrics_raw.items()},
            runtime_ms=None if runtime is None else float(runtime),
            warnings=tuple(str(w) for w in warnings_raw),
        )

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _describe_transport_failure(url: str, exc: requests.RequestException) -> str:
        return (
            f"Could not reach the backend at {url} ({type(exc).__name__}). "
            "Check that the API is running, that the host/IP is correct for this "
            "machine, and that the port is open."
        )

    @staticmethod
    def _decode(response: requests.Response, url: str) -> Mapping[str, Any]:
        if response.status_code >= 400:
            detail = response.text.strip()[:300] or "<empty body>"
            raise ApiResponseError(f"{url} returned HTTP {response.status_code}: {detail}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiResponseError(f"{url} did not return valid JSON.") from exc
        if not isinstance(payload, Mapping):
            raise ApiResponseError(f"{url} returned JSON of type {type(payload).__name__}.")
        return payload
