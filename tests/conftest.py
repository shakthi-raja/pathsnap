"""Shared pytest fixtures.

The API-client tests exercise the real ``PathSnapClient`` against a fake
transport, so no test ever opens a socket or depends on a running backend.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from src.config import Settings


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, payload: Any, status_code: int = 200, *, text: str | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Records calls and returns queued responses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response: Any = None
        self._exception: Exception | None = None

    def queue(self, response: Any) -> FakeSession:
        self._response = response
        return self

    def fail_with(self, exception: Exception) -> FakeSession:
        self._exception = exception
        return self

    def _dispatch(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": method, "url": url, **kwargs})
        if self._exception is not None:
            raise self._exception
        return self._response

    def get(self, url: str, **kwargs: Any) -> Any:
        return self._dispatch("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        return self._dispatch("POST", url, **kwargs)

    def close(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing at a throwaway data root."""
    data_root = tmp_path / "raw"
    (data_root / "train").mkdir(parents=True)
    (data_root / "student_test").mkdir(parents=True)
    return Settings(
        api_url="http://127.0.0.1:8000",
        api_timeout_s=30.0,
        health_timeout_s=5.0,
        data_root=data_root,
    )


@pytest.fixture
def fake_session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def response_factory() -> Callable[..., FakeResponse]:
    return FakeResponse
