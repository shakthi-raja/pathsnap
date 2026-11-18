"""Tests for environment-driven configuration."""

from __future__ import annotations

import pytest

from src import config
from src.config import ConfigurationError, get_settings


@pytest.fixture(autouse=True)
def _clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_explicit_data_root_is_used(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "raw"
    data_root.mkdir()
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(data_root))

    settings = get_settings()

    assert settings.data_root == data_root.resolve()


def test_missing_data_root_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path / "does-not-exist"))

    with pytest.raises(ConfigurationError, match="Could not locate the raw dataset"):
        get_settings()


def test_api_url_is_normalised(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "raw"
    data_root.mkdir()
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PATHSNAP_API_URL", "http://192.168.1.5:8000/")

    assert get_settings().api_url == "http://192.168.1.5:8000"


def test_api_url_requires_scheme(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "raw"
    data_root.mkdir()
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PATHSNAP_API_URL", "192.168.1.5:8000")

    with pytest.raises(ConfigurationError, match="must start with http"):
        get_settings()


@pytest.mark.parametrize("value", ["0", "-5", "abc"])
def test_timeout_must_be_positive_number(tmp_path, monkeypatch, value) -> None:
    data_root = tmp_path / "raw"
    data_root.mkdir()
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PATHSNAP_API_TIMEOUT_S", value)

    with pytest.raises(ConfigurationError):
        get_settings()


def test_settings_are_cached() -> None:
    assert get_settings.cache_info().maxsize == 1
    assert config._REPO_ROOT.is_dir()
