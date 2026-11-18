"""Single source of truth for map styling.

Colours are chosen for colour-vision accessibility and sufficient contrast on
the default OpenStreetMap basemap. Every layer name exposed here is also the
name shown in the Folium layer control and in the legend, so the two can never
drift apart.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "GPS_LAYER",
    "GPS_MARKER_COLOUR",
    "GPS_MARKER_RADIUS",
    "MAX_GPS_MARKERS",
    "PREDICTED_LAYER",
    "PREDICTED_WEIGHT",
    "PREDICTED_COLOUR",
    "REFERENCE_LAYER",
    "REFERENCE_COLOUR",
    "ROAD_LAYER",
    "ROAD_COLOUR",
    "ROAD_OPACITY",
    "ROAD_WEIGHT",
]

# --- Raw GPS observations -------------------------------------------------
GPS_LAYER: Final[str] = "Raw GPS"
GPS_MARKER_COLOUR: Final[str] = "#2563EB"  # blue-600
GPS_MARKER_RADIUS: Final[int] = 2

#: Upper bound on rendered GPS markers. A 19k-point trip would otherwise
#: produce a multi-megabyte map that is unusable in a browser.
MAX_GPS_MARKERS: Final[int] = 600

# --- Road network ---------------------------------------------------------
ROAD_LAYER: Final[str] = "Road network"
ROAD_COLOUR: Final[str] = "#94A3B8"  # slate-400
ROAD_WEIGHT: Final[int] = 2
ROAD_OPACITY: Final[float] = 0.45

# --- Predicted route ------------------------------------------------------
PREDICTED_LAYER: Final[str] = "Predicted route"
PREDICTED_COLOUR: Final[str] = "#16A34A"  # green-600
PREDICTED_WEIGHT: Final[int] = 5

# --- Reference route (development / report only) --------------------------
REFERENCE_LAYER: Final[str] = "Reference route"
REFERENCE_COLOUR: Final[str] = "#EA580C"  # orange-600
