"""Tests for the typed HTTP client."""

from __future__ import annotations

import pytest
import requests

from src.config import Settings
from src.ui.api_client import ApiResponseError, ApiUnavailable, PathSnapClient


def _client(settings: Settings, session) -> PathSnapClient:
    return PathSnapClient(settings, session=session)


def test_health_returns_payload(settings: Settings, fake_session, response_factory) -> None:
    fake_session.queue(response_factory({"status": "ok"}))

    assert _client(settings, fake_session).health() == {"status": "ok"}
    assert fake_session.calls[0]["url"] == "http://127.0.0.1:8000/health"


def test_list_trips_returns_tuple(settings: Settings, fake_session, response_factory) -> None:
    fake_session.queue(response_factory({"trip_ids": ["00000000", "00000007"]}))

    assert _client(settings, fake_session).list_trips() == ("00000000", "00000007")


def test_list_trips_rejects_non_string_ids(settings, fake_session, response_factory) -> None:
    fake_session.queue(response_factory({"trip_ids": [1, 2]}))

    with pytest.raises(ApiResponseError, match="must contain strings"):
        _client(settings, fake_session).list_trips()


def test_predict_parses_full_payload(settings, fake_session, response_factory) -> None:
    payload = {
        "trip_id": "00000065",
        "status": "ok",
        "track_points": [[53.5, 49.3], [53.6, 49.4]],
        "road_context": [[[53.5, 49.3], [53.6, 49.4]]],
        "predicted_polylines": [[[53.5, 49.3], [53.6, 49.4]]],
        "metrics": {"pointwise_accuracy": 0.9, "route_continuity": None},
        "runtime_ms": 1234.0,
        "warnings": ["low candidate density"],
    }
    fake_session.queue(response_factory(payload))

    prediction = _client(settings, fake_session).predict("00000065")

    assert prediction.trip_id == "00000065"
    assert prediction.status == "ok"
    assert prediction.track_points == ((53.5, 49.3), (53.6, 49.4))
    assert prediction.metrics["route_continuity"] is None
    assert prediction.runtime_ms == 1234.0
    assert prediction.warnings == ("low candidate density",)
    assert fake_session.calls[0]["json"] == {"trip_id": "00000065"}


def test_predict_rejects_blank_trip_id(settings, fake_session) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _client(settings, fake_session).predict("   ")


def test_http_error_becomes_response_error(settings, fake_session, response_factory) -> None:
    fake_session.queue(response_factory({"detail": "boom"}, status_code=404, text="not found"))

    with pytest.raises(ApiResponseError, match="HTTP 404"):
        _client(settings, fake_session).predict("00000099")


def test_connection_error_becomes_unavailable(settings, fake_session) -> None:
    fake_session.fail_with(requests.ConnectionError("refused"))

    with pytest.raises(ApiUnavailable, match="Could not reach the backend"):
        _client(settings, fake_session).health()


def test_missing_required_key_is_reported(settings, fake_session, response_factory) -> None:
    fake_session.queue(response_factory({"trip_id": "x"}))

    with pytest.raises(ApiResponseError, match="missing required key 'status'"):
        _client(settings, fake_session).predict("x")


def test_invalid_json_becomes_response_error(settings, fake_session, response_factory) -> None:
    fake_session.queue(response_factory(None, status_code=200))

    with pytest.raises(ApiResponseError, match="did not return valid JSON"):
        _client(settings, fake_session).health()


def test_client_does_not_close_injected_session(settings, fake_session) -> None:
    with _client(settings, fake_session):
        pass
    assert fake_session.calls == []
