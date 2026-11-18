
from pathlib import Path
import re
import pandas as pd


def normalize_trip_id(trip_id):
    """Accept one to eight ASCII digits; never permit path components as IDs."""
    value = str(trip_id)
    if re.fullmatch(r'[0-9]{1,8}', value) is None:
        raise ValueError('trip_id must contain one to eight digits')
    return value.zfill(8)


def _read_table(path, columns):
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(columns) or frame.empty:
        raise ValueError(f"Expected {len(columns)} columns and nonempty rows: {path}")
    frame.columns = columns
    return frame.apply(pd.to_numeric, errors='raise')


def load_trip(dataset_folder, trip_id, *, include_route=True, validate=True):
    """
    Load GPS observations, road nodes,
    road arcs and the reference route.
    """

    trip_id = normalize_trip_id(trip_id)
    folder = Path(dataset_folder) / trip_id

    if not folder.exists():
        raise FileNotFoundError(f"Trip folder missing: {folder}")

    track = _read_table(folder / f"{trip_id}.track", ["longitude", "latitude", "timestamp"])

    nodes = _read_table(folder / f"{trip_id}.nodes", ["longitude", "latitude"])

    arcs = _read_table(folder / f"{trip_id}.arcs", ["from_node_id", "to_node_id"])

    route = _read_table(folder / f"{trip_id}.route", ["arc_id"]) if include_route else None

    # Original row numbers represent IDs.
    track.index.name = "gps_index"
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    if not validate:
        return track, nodes, arcs, route

    # Basic data validation.
    if not track["longitude"].between(-180, 180).all():
        raise ValueError("Invalid GPS longitude")

    if not track["latitude"].between(-90, 90).all():
        raise ValueError("Invalid GPS latitude")

    if not (
        arcs[["from_node_id", "to_node_id"]]
        .ge(0).all().all()
        and arcs[["from_node_id", "to_node_id"]]
        .lt(len(nodes)).all().all()
    ):
        raise ValueError("Invalid node ID in arcs")

    if route is not None and not route["arc_id"].between(
        0, len(arcs) - 1
    ).all():
        raise ValueError("Invalid arc ID in reference route")

    if not nodes.longitude.between(-180, 180).all() or not nodes.latitude.between(-90, 90).all():
        raise ValueError("Invalid road-node coordinates")
    import numpy as np
    if not np.isfinite(track.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite GPS coordinates or timestamps")
    if not np.isfinite(nodes.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite node coordinates")
    if not np.equal(arcs.to_numpy(), np.floor(arcs.to_numpy())).all():
        raise ValueError("Non-integer node reference")
    if route is not None and not np.equal(route.to_numpy(), np.floor(route.to_numpy())).all():
        raise ValueError("Non-integer route reference")

    return track, nodes, arcs, route


if __name__ == "__main__":
    track, nodes, arcs, route = load_trip(
        "data/raw/train",
        "00000001"
    )

    print("\n========== PATHSNAP DATASET ==========")

    print("\nGPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Reference route arcs:", len(route))

    print("\n--- GPS observations ---")
    print(track.head())

    print("\n--- Road nodes ---")
    print(nodes.head())

    print("\n--- Road arcs ---")
    print(arcs.head())

    print("\n--- Reference route ---")
    print(route.head())

    print("\nDataset loaded successfully!")
