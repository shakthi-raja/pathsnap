"""Application settings.

Configuration is resolved from the environment in exactly one place so that
defaults, validation and error messages live together. Modules must not call
``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["ConfigurationError", "Settings", "get_settings"]

#: Repository root (``<repo>/src/config.py`` -> ``<repo>``).
_REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_API_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_HEALTH_TIMEOUT_S = 5.0


class ConfigurationError(RuntimeError):
    """Raised when the application cannot resolve a required setting."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, validated application settings.

    Attributes:
        api_url: Base URL of the PathSnap FastAPI backend (no trailing slash).
        api_timeout_s: Timeout for prediction requests, in seconds.
        health_timeout_s: Timeout for the lightweight health probe, in seconds.
        data_root: Directory holding the ``train/`` and ``student_test/``
            partitions of the raw dataset.
    """

    api_url: str
    api_timeout_s: float
    health_timeout_s: float
    data_root: Path


def _resolve_api_url() -> str:
    raw = os.environ.get("PATHSNAP_API_URL", _DEFAULT_API_URL).strip()
    if not raw:
        raise ConfigurationError("PATHSNAP_API_URL is set but empty.")
    if not raw.startswith(("http://", "https://")):
        raise ConfigurationError(
            f"PATHSNAP_API_URL must start with http:// or https://, got {raw!r}."
        )
    return raw.rstrip("/")


def _resolve_timeout(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric, got {raw!r}.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}.")
    return value


def _resolve_data_root() -> Path:
    """Locate the raw dataset, preferring an explicit override.

    The raw data deliberately lives outside the repository so that a 91 MB
    dataset is never committed. Deployments must therefore point
    ``PATHSNAP_DATA_ROOT`` at it; locally we fall back to the two conventional
    sibling locations.
    """
    override = os.environ.get("PATHSNAP_DATA_ROOT")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    else:
        candidates.extend((_REPO_ROOT / "data" / "raw", _REPO_ROOT.parent / "data" / "raw"))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    tried = "\n  ".join(str(c) for c in candidates) or "(none)"
    raise ConfigurationError(
        "Could not locate the raw dataset. Set PATHSNAP_DATA_ROOT to the "
        f"directory containing train/ and student_test/. Tried:\n  {tried}"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, resolved on first use.

    Cached so that repeated calls from Streamlit reruns stay cheap. Tests can
    clear the cache with ``get_settings.cache_clear()``.
    """
    return Settings(
        api_url=_resolve_api_url(),
        api_timeout_s=_resolve_timeout("PATHSNAP_API_TIMEOUT_S", _DEFAULT_TIMEOUT_S),
        health_timeout_s=_resolve_timeout("PATHSNAP_HEALTH_TIMEOUT_S", _DEFAULT_HEALTH_TIMEOUT_S),
        data_root=_resolve_data_root(),
    )
