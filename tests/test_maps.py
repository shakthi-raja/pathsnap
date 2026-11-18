"""Tests for map construction and payload validation."""

from __future__ import annotations

import folium
import pytest

from src.visualization import palette
from src.visualization.maps import (
    InvalidGeometryError,
    _downsample,
    build_trip_map,
    build_trip_map_from_files,
    normalise_payload,
)

_BASE_PAYLOAD = {
    "track_points": [[52.35, 4.90], [52.36, 4.91], [52.37, 4.92]],
    "road_context": [[[52.35, 4.90], [52.36, 4.91]]],
    "predicted_polylines": [[[52.35, 4.90], [52.36, 4.91], [52.37, 4.92]]],
}


def test_normalise_payload_converts_to_tuples() -> None:
    payload = normalise_payload(_BASE_PAYLOAD)

    assert payload.track_points[0] == (52.35, 4.90)
    assert isinstance(payload.track_points, tuple)
    assert payload.reference_polylines == ()


def test_normalise_payload_missing_key_raises() -> None:
    broken = {"track_points": []}

    with pytest.raises(KeyError, match="predicted_polylines"):
        normalise_payload(broken)


@pytest.mark.parametrize(
    "point",
    [
        [91.0, 4.9],
        [-91.0, 4.9],
        [52.0, 181.0],
        [52.0, -181.0],
        [float("nan"), 4.9],
        [52.0, float("inf")],
        ["a", "b"],
        [52.0],
        "not-a-point",
    ],
)
def test_invalid_coordinates_are_rejected(point) -> None:
    with pytest.raises(InvalidGeometryError):
        normalise_payload({**_BASE_PAYLOAD, "track_points": [point]})


def test_build_trip_map_contains_all_layers() -> None:
    trip_map = build_trip_map(normalise_payload(_BASE_PAYLOAD))
    html = trip_map.get_root().render()

    assert isinstance(trip_map, folium.Map)
    for layer in (
        palette.GPS_LAYER,
        palette.ROAD_LAYER,
        palette.PREDICTED_LAYER,
    ):
        assert layer in html
    assert palette.REFERENCE_LAYER not in html


def test_reference_layer_only_when_requested() -> None:
    payload = normalise_payload(
        _BASE_PAYLOAD,
        reference_polylines=[[[52.35, 4.90], [52.36, 4.91]]],
    )
    html = build_trip_map(payload).get_root().render()

    assert palette.REFERENCE_LAYER in html
    assert "Reference route" in html


def test_empty_payload_still_renders_a_map() -> None:
    payload = normalise_payload({"track_points": [], "road_context": [], "predicted_polylines": []})
    trip_map = build_trip_map(payload)

    assert isinstance(trip_map, folium.Map)


def test_downsample_respects_limit_and_keeps_endpoints() -> None:
    points = tuple((float(i), 0.0) for i in range(1000))

    sampled = _downsample(points, 100)

    assert len(sampled) <= 100
    assert sampled[0] == points[0]


def test_downsample_noop_when_under_limit() -> None:
    points = ((1.0, 2.0), (3.0, 4.0))

    assert _downsample(points, 100) == points


def test_single_point_track_does_not_crash() -> None:
    payload = normalise_payload(
        {"track_points": [[52.35, 4.90]], "road_context": [], "predicted_polylines": []}
    )

    assert isinstance(build_trip_map(payload), folium.Map)


def test_build_from_files_reads_raw_trip(tmp_path) -> None:
    stem = "00000099"
    trip = tmp_path / "train" / stem
    trip.mkdir(parents=True)
    # Source order is lon lat; the map must flip to lat lon.
    (trip / f"{stem}.track").write_text(
        "4.90\t52.35\t1.0\r\n4.91\t52.36\t2.0\r\n", encoding="ascii"
    )
    (trip / f"{stem}.nodes").write_text("4.90\t52.35\n4.91\t52.36\n4.92\t52.37\n", encoding="ascii")
    (trip / f"{stem}.arcs").write_text("0\t1\n1\t2\n", encoding="ascii")
    (trip / f"{stem}.route").write_text("0\n1\n", encoding="ascii")

    trip_map = build_trip_map_from_files(
        track_path=str(trip / f"{stem}.track"),
        nodes_path=str(trip / f"{stem}.nodes"),
        arcs_path=str(trip / f"{stem}.arcs"),
        route_path=str(trip / f"{stem}.route"),
    )
    html = trip_map.get_root().render()

    assert "52.35" in html
    assert palette.REFERENCE_LAYER in html
