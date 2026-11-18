"""Assemble the per-trip candidate tables exported by :mod:`src.data.pipeline`.

The data pipeline writes one parquet per trip plus an alignment sidecar. Models
consume a single concatenated frame, so the assembly and the small amount of
schema translation live here rather than in every training script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["list_trip_ids", "load_split", "labelled_rows", "model_frame"]

TRIP_ID_LENGTH = 8


def list_trip_ids(directory: str | Path) -> list[str]:
    """Return trip IDs that have a candidate table in ``directory``.

    Alignment sidecars (``<trip>.alignment.parquet``) are excluded because
    their stem is not an eight digit trip ID.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"No processed data directory at {directory}")
    trip_ids = sorted(
        path.stem
        for path in directory.glob("*.parquet")
        if path.stem.isdigit() and len(path.stem) == TRIP_ID_LENGTH
    )
    if not trip_ids:
        raise FileNotFoundError(f"No candidate tables in {directory}. Run src.data.pipeline first.")
    return trip_ids


def load_split(directory: str | Path) -> pd.DataFrame:
    """Concatenate every trip's candidate table, tagging each row with its trip."""
    directory = Path(directory)
    frames = []
    for trip_id in list_trip_ids(directory):
        frame = pd.read_parquet(directory / f"{trip_id}.parquet")
        frame["trip_id"] = trip_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def labelled_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows the alignment marked confident and that have a label.

    Everything else has no trustworthy target and must stay out of training and
    scoring. The alignment deliberately withholds labels where the reference
    route is ambiguous, so dropping them is the intended behaviour rather than
    a data loss.
    """
    eligible = data["training_eligible"].fillna(False).astype(bool)
    has_label = data["label"].notna()
    return data.loc[eligible & has_label].copy()


def model_frame(directory: str | Path) -> pd.DataFrame:
    """Load a split and return labelled rows with the columns models expect.

    ``obs_idx`` is the observation identifier used by the metrics helpers.
    """
    frame = labelled_rows(load_split(directory))
    frame = frame.rename(columns={"gps_index": "obs_idx"})
    frame["label"] = frame["label"].astype(int)
    return frame
