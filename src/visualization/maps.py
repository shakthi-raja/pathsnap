"""Folium map construction.

This module is deliberately free of Streamlit and of the raw dataset format: it
consumes the API payload defined by the team contract and returns a plain
``folium.Map``. That keeps map rendering unit-testable without a browser or a
running backend.

Coordinate convention
---------------------
The API contract returns ``[latitude, longitude]`` (display order). Raw source
files store longitude first, so conversion happens before this module is
reached. The swap happens before this module.

Missing data
------------
The API returns ``predicted_polylines`` as a list of polylines rather than one
polyline so that GPS gaps stay gaps. Segments are never joined with a
fictitious straight line.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import folium

from . import palette

__all__ = [
    "InvalidGeometryError",
    "normalise_payload",
    "build_trip_map",
    "build_trip_map_from_files",
]

Coordinate = tuple[float, float]

_LAT_RANGE: Final[tuple[float, float]] = (-90.0, 90.0)
_LON_RANGE: Final[tuple[float, float]] = (-180.0, 180.0)
_DEFAULT_CENTRE: Final[Coordinate] = (0.0, 0.0)
_DEFAULT_ZOOM: Final[int] = 2
_FIT_ZOOM: Final[int] = 14


class InvalidGeometryError(ValueError):
    """Raised when a coordinate cannot be placed on a map."""


@dataclass(frozen=True, slots=True)
class MapPayload:
    """Validated geometry ready for rendering.

    Attributes:
        track_points: Raw GPS observations as ``(lat, lon)``.
        road_context: Road-network segments, each a list of ``(lat, lon)``.
        predicted_polylines: Reconstructed route pieces. Multiple pieces mean
            the trajectory had gaps that must not be visually bridged.
        reference_polylines: Optional reference route, for development and
            report maps only. Never populated from the evaluator partition.
    """

    track_points: tuple[Coordinate, ...]
    road_context: tuple[tuple[Coordinate, ...], ...]
    predicted_polylines: tuple[tuple[Coordinate, ...], ...]
    reference_polylines: tuple[tuple[Coordinate, ...], ...] = ()


def _as_coordinate(value: Any, *, field: str) -> Coordinate:
    """Validate one ``[lat, lon]`` pair and return it as a float tuple."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvalidGeometryError(f"{field}: expected a [lat, lon] pair, got {value!r}")
    if len(value) != 2:
        raise InvalidGeometryError(f"{field}: expected 2 values, got {len(value)}")
    try:
        lat, lon = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise InvalidGeometryError(f"{field}: non-numeric coordinates {value!r}") from exc
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise InvalidGeometryError(f"{field}: non-finite coordinates {value!r}")
    if not _LAT_RANGE[0] <= lat <= _LAT_RANGE[1]:
        raise InvalidGeometryError(f"{field}: latitude {lat} outside [-90, 90]")
    if not _LON_RANGE[0] <= lon <= _LON_RANGE[1]:
        raise InvalidGeometryError(f"{field}: longitude {lon} outside [-180, 180]")
    return lat, lon


def _as_line(value: Any, *, field: str) -> tuple[Coordinate, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvalidGeometryError(f"{field}: expected a list of [lat, lon] pairs")
    return tuple(_as_coordinate(point, field=f"{field}[{i}]") for i, point in enumerate(value))


def _as_lines(value: Any, *, field: str) -> tuple[tuple[Coordinate, ...], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvalidGeometryError(f"{field}: expected a list of polylines")
    return tuple(_as_line(line, field=f"{field}[{i}]") for i, line in enumerate(value))


def normalise_payload(
    payload: Mapping[str, Any],
    *,
    reference_polylines: Iterable[Sequence[Sequence[float]]] | None = None,
) -> MapPayload:
    """Validate an API response and convert it into a :class:`MapPayload`.

    Args:
        payload: A ``POST /predict`` response body.
        reference_polylines: Optional reference geometry. Supplied only by
            development/report tooling; never by the evaluator UI.

    Raises:
        InvalidGeometryError: If any coordinate is malformed or out of range.
        KeyError: If a required key is missing from ``payload``.
    """
    missing = {"track_points", "road_context", "predicted_polylines"} - payload.keys()
    if missing:
        raise KeyError(f"API response is missing required keys: {sorted(missing)}")

    return MapPayload(
        track_points=_as_line(payload["track_points"], field="track_points"),
        road_context=_as_lines(payload["road_context"], field="road_context"),
        predicted_polylines=_as_lines(payload["predicted_polylines"], field="predicted_polylines"),
        reference_polylines=(
            _as_lines(list(reference_polylines), field="reference_polylines")
            if reference_polylines
            else ()
        ),
    )


def _downsample(points: Sequence[Coordinate], limit: int) -> Sequence[Coordinate]:
    """Return at most ``limit`` evenly spaced points.

    Sampling (rather than slicing) keeps the visual density of the trajectory
    representative while bounding the size of the generated HTML.
    """
    if limit <= 0 or len(points) <= limit:
        return points
    stride = math.ceil(len(points) / limit)
    return points[::stride]


def _road_feature_collection(
    segments: Sequence[Sequence[Coordinate]],
) -> dict[str, Any]:
    """Package road segments as a single GeoJSON ``FeatureCollection``.

    GeoJSON mandates ``[longitude, latitude]`` order, the opposite of our
    internal ``(lat, lon)`` coordinates, so the swap happens here and nowhere
    else.
    """
    features = [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lat, lon in segment],
            },
        }
        for segment in segments
        if len(segment) >= 2
    ]
    return {"type": "FeatureCollection", "features": features}


def _legend_html(entries: Sequence[tuple[str, str]]) -> str:
    """Build a static legend for the layers actually present on the map.

    All values are module constants, never user input.
    """
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{colour};margin-right:6px;border-radius:2px;"></span>'
        f"<span>{label}</span></div>"
        for colour, label in entries
    )
    return (
        '<div style="position:fixed;bottom:28px;left:28px;z-index:9999;'
        "background:rgba(255,255,255,0.92);padding:10px 12px;"
        "border:1px solid #CBD5E1;border-radius:8px;"
        "font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;color:#0F172A;"
        'box-shadow:0 2px 6px rgba(15,23,42,0.15);">'
        '<div style="font-weight:600;margin-bottom:6px;">Legend</div>'
        f"{rows}</div>"
    )


def build_trip_map(
    payload: MapPayload,
    *,
    zoom_start: int = _FIT_ZOOM,
    max_gps_markers: int = palette.MAX_GPS_MARKERS,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """Render a validated payload as an interactive Folium map.

    Args:
        payload: Geometry produced by :func:`normalise_payload`.
        zoom_start: Initial zoom when the payload has no geometry to fit.
        max_gps_markers: Cap on rendered GPS markers.
        tiles: Folium tile set name.

    Returns:
        A configured ``folium.Map`` with layer control and legend attached.
    """
    centre = payload.track_points[0] if payload.track_points else _DEFAULT_CENTRE
    trip_map = folium.Map(
        location=centre,
        zoom_start=zoom_start if payload.track_points else _DEFAULT_ZOOM,
        tiles=tiles,
        control_scale=True,
    )

    if payload.road_context:
        # A city extract can contain tens of thousands of segments. Emitting one
        # Leaflet layer per segment bloats the page by an order of magnitude, so
        # the whole context is sent as a single GeoJSON feature collection.
        roads = folium.FeatureGroup(name=palette.ROAD_LAYER, show=True)
        folium.GeoJson(
            _road_feature_collection(payload.road_context),
            style_function=lambda _: {
                "color": palette.ROAD_COLOUR,
                "weight": palette.ROAD_WEIGHT,
                "opacity": palette.ROAD_OPACITY,
            },
            name=palette.ROAD_LAYER,
        ).add_to(roads)
        roads.add_to(trip_map)

    # Draw order matters: observations sit underneath the reconstructed routes,
    # otherwise a dense GPS cloud hides the line the evaluator came to see.
    if payload.track_points:
        gps = folium.FeatureGroup(name=palette.GPS_LAYER, show=True)
        for lat, lon in _downsample(payload.track_points, max_gps_markers):
            folium.CircleMarker(
                location=(lat, lon),
                radius=palette.GPS_MARKER_RADIUS,
                color=palette.GPS_MARKER_COLOUR,
                fill=True,
                fill_color=palette.GPS_MARKER_COLOUR,
                fill_opacity=0.7,
                weight=1,
            ).add_to(gps)
        gps.add_to(trip_map)

    if payload.reference_polylines:
        reference = folium.FeatureGroup(name=palette.REFERENCE_LAYER, show=True)
        for line in payload.reference_polylines:
            if len(line) < 2:
                continue
            folium.PolyLine(
                locations=line,
                color=palette.REFERENCE_COLOUR,
                weight=4,
                opacity=0.9,
            ).add_to(reference)
        reference.add_to(trip_map)

    if payload.predicted_polylines:
        predicted = folium.FeatureGroup(name=palette.PREDICTED_LAYER, show=True)
        for line in payload.predicted_polylines:
            if len(line) < 2:
                continue
            folium.PolyLine(
                locations=line,
                color=palette.PREDICTED_COLOUR,
                weight=palette.PREDICTED_WEIGHT,
                opacity=0.9,
            ).add_to(predicted)
        predicted.add_to(trip_map)

    if payload.track_points:
        folium.Marker(
            payload.track_points[0], tooltip="Start", icon=folium.Icon(color="blue")
        ).add_to(trip_map)
        folium.Marker(
            payload.track_points[-1], tooltip="End", icon=folium.Icon(color="red")
        ).add_to(trip_map)
        trip_map.fit_bounds(
            [
                [min(p[0] for p in payload.track_points), min(p[1] for p in payload.track_points)],
                [max(p[0] for p in payload.track_points), max(p[1] for p in payload.track_points)],
            ]
        )

    legend_entries: list[tuple[str, str]] = []
    if payload.track_points:
        legend_entries.append((palette.GPS_MARKER_COLOUR, palette.GPS_LAYER))
    if payload.road_context:
        legend_entries.append((palette.ROAD_COLOUR, palette.ROAD_LAYER))
    if payload.predicted_polylines:
        legend_entries.append((palette.PREDICTED_COLOUR, palette.PREDICTED_LAYER))
    if payload.reference_polylines:
        legend_entries.append((palette.REFERENCE_COLOUR, palette.REFERENCE_LAYER))

    folium.LayerControl(collapsed=False).add_to(trip_map)
    trip_map.get_root().html.add_child(folium.Element(_legend_html(legend_entries)))
    return trip_map


def build_trip_map_from_files(
    track_path: str,
    nodes_path: str,
    arcs_path: str,
    route_path: str | None = None,
    *,
    max_road_segments: int = 15_000,
    max_gps_markers: int = palette.MAX_GPS_MARKERS,
) -> folium.Map:
    """Render a report/EDA map straight from raw dataset files.

    Used to produce the Folium evidence required by the report on **training or
    student-test** trips, where the reference route is permitted. The evaluator
    UI must never call this.

    Args:
        track_path: Path to ``<trip>.track`` (lon, lat, timestamp).
        nodes_path: Path to ``<trip>.nodes`` (lon, lat).
        arcs_path: Path to ``<trip>.arcs`` (from_node, to_node).
        route_path: Optional path to ``<trip>.route`` (arc ids).
        max_road_segments: Deterministic cap on rendered road segments.
        max_gps_markers: Cap on rendered GPS markers.
    """
    track = _read_track(track_path)
    nodes = _read_nodes(nodes_path)
    arcs = _read_arcs(arcs_path, node_count=len(nodes))

    payload = MapPayload(
        track_points=track,
        road_context=_select_road_context(nodes, arcs, track, limit=max_road_segments),
        predicted_polylines=(),
        reference_polylines=(_reference_polylines(nodes, arcs, route_path),) if route_path else (),
    )
    return build_trip_map(payload, max_gps_markers=max_gps_markers)


# --------------------------------------------------------------------------
# Raw-file readers (report tooling only)
# --------------------------------------------------------------------------


def _read_track(path: str) -> tuple[Coordinate, ...]:
    points: list[Coordinate] = []
    with open(path, encoding="ascii", newline="") as handle:
        for lineno, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected at least 2 columns")
            # Source order is lon, lat; the map needs lat, lon.
            points.append(_as_coordinate((parts[1], parts[0]), field=f"{path}:{lineno}"))
    return tuple(points)


def _read_nodes(path: str) -> list[Coordinate]:
    nodes: list[Coordinate] = []
    with open(path, encoding="ascii", newline="") as handle:
        for lineno, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            nodes.append(_as_coordinate((parts[1], parts[0]), field=f"{path}:{lineno}"))
    return nodes


def _read_arcs(path: str, *, node_count: int) -> list[tuple[int, int]]:
    arcs: list[tuple[int, int]] = []
    with open(path, encoding="ascii", newline="") as handle:
        for lineno, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                source, target = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: non-integer node ids") from exc
            if not (0 <= source < node_count and 0 <= target < node_count):
                raise ValueError(f"{path}:{lineno}: node id outside 0..{node_count - 1}")
            arcs.append((source, target))
    return arcs


def _select_road_context(
    nodes: Sequence[Coordinate],
    arcs: Sequence[tuple[int, int]],
    track: Sequence[Coordinate],
    *,
    limit: int,
) -> tuple[tuple[Coordinate, ...], ...]:
    """Keep road segments near the trajectory, capped for browser performance.

    A whole-city network (up to 307k arcs) cannot be embedded in an HTML map.
    We filter to the trajectory's bounding box and then take an even stride,
    which preserves overall structure while bounding output size.
    """
    if not track or not arcs:
        return ()

    pad = 0.01  # degrees, roughly 1 km
    lat_min = min(p[0] for p in track) - pad
    lat_max = max(p[0] for p in track) + pad
    lon_min = min(p[1] for p in track) - pad
    lon_max = max(p[1] for p in track) + pad

    def inside(point: Coordinate) -> bool:
        return lat_min <= point[0] <= lat_max and lon_min <= point[1] <= lon_max

    segments = [
        (nodes[source], nodes[target])
        for source, target in arcs
        if inside(nodes[source]) or inside(nodes[target])
    ]
    if limit > 0 and len(segments) > limit:
        stride = math.ceil(len(segments) / limit)
        segments = segments[::stride]

    return tuple((a, b) for a, b in segments)


def _reference_polylines(
    nodes: Sequence[Coordinate],
    arcs: Sequence[tuple[int, int]],
    route_path: str,
) -> tuple[Coordinate, ...]:
    """Polyline for the reference route, split where arcs are discontinuous."""
    with open(route_path, encoding="ascii", newline="") as handle:
        route = [int(tokens[0]) for line in handle if (tokens := line.split())]

    for index in route:
        if not 0 <= index < len(arcs):
            raise ValueError(f"{route_path}: arc id {index} outside 0..{len(arcs) - 1}")

    points: list[Coordinate] = []
    previous_target: int | None = None
    for arc_id in route:
        source, target = arcs[arc_id]
        if previous_target is not None and source != previous_target:
            break  # discontinuous reference: render the continuous prefix only
        if not points:
            points.append(nodes[source])
        points.append(nodes[target])
        previous_target = target
    return tuple(points)
