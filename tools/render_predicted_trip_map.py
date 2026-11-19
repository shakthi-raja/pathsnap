"""Render a report map that shows the predicted route beside the reference route.

The evaluator interface never sees ground truth, so this tool exists only for
the technical report: it runs the trained pipeline on a training or
student-testing trip, takes the predicted geometry from the service, and adds
the reference route for comparison. The resulting HTML carries four togglable
layers (road network, GPS observations, predicted route, reference route) and a
static legend.

Usage::

    python tools/render_predicted_trip_map.py --trip 00000062 \\
        --partition train --data-root ../data/raw \\
        --out-html results/maps/trip_00000062_layers.html \\
        --out-png report/figures/folium_trip_00000062_layers.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python tools/render_predicted_trip_map.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.service import PathSnapService  # noqa: E402
from src.visualization import palette  # noqa: E402
from src.visualization.maps import (  # noqa: E402
    _read_arcs,
    _read_nodes,
    _reference_polylines,
    build_trip_map,
    normalise_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", required=True, help="Trip id, e.g. 00000062")
    parser.add_argument(
        "--partition", default="train", choices=("train", "student_test"), help="Dataset partition"
    )
    parser.add_argument("--data-root", type=Path, default=Path("../data/raw"))
    parser.add_argument(
        "--model-path", type=Path, default=PROJECT_ROOT / "artifacts" / "final_pipeline.joblib"
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=PROJECT_ROOT / "artifacts" / "model_metadata.json"
    )
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--max-road-segments", type=int, default=12_000)
    return parser.parse_args(argv)


def build_map(args: argparse.Namespace):
    """Run the pipeline once and assemble the four-layer map."""
    partition_root = args.data_root / args.partition
    service = PathSnapService(
        evaluation_root=partition_root,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
    )
    result = service.predict(args.trip)

    trip_dir = partition_root / args.trip
    route_path = trip_dir / f"{args.trip}.route"
    reference = ()
    if route_path.is_file():
        nodes = _read_nodes(str(trip_dir / f"{args.trip}.nodes"))
        arcs = _read_arcs(str(trip_dir / f"{args.trip}.arcs"), node_count=len(nodes))
        reference = (_reference_polylines(nodes, arcs, str(route_path)),)

    payload = normalise_payload(
        {
            "track_points": result.track_points,
            "road_context": result.road_context,
            "predicted_polylines": result.predicted_polylines,
        },
        reference_polylines=reference,
    )
    trip_map = build_trip_map(payload)
    _dash_reference(trip_map)
    return trip_map


def _dash_reference(trip_map) -> None:
    """Draw the reference as a dashed line on top of the solid prediction.

    The two routes agree on most arcs, so the reference is thinned and dashed
    and moved to the end of the draw order. Where the lines overlap the dashes
    still read as a second layer, and where they diverge both are visible.
    """
    children = trip_map._children
    reference_key = next(
        key
        for key, child in children.items()
        if getattr(child, "layer_name", None) == palette.REFERENCE_LAYER
    )
    reference = children[reference_key]
    for line in reference._children.values():
        if hasattr(line, "options"):
            line.options["dashArray"] = "7 5"
            line.options["weight"] = 3
    items = [(key, child) for key, child in children.items() if key != reference_key]
    control = next(
        ((key, child) for key, child in items if type(child).__name__ == "LayerControl"),
        None,
    )
    if control is not None:
        items = [(key, child) for key, child in items if key != control[0]]
        items.extend([(reference_key, reference), control])
    else:
        items.append((reference_key, reference))
    trip_map._children = dict(items)


def screenshot(html_path: Path, png_path: Path, *, width: int = 1280, height: int = 760) -> None:
    """Capture the rendered map with headless Chromium."""
    from playwright.sync_api import sync_playwright

    png_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
    executable = str(candidates[-1]) if candidates else None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(html_path.resolve().as_uri())
        page.wait_for_timeout(4500)
        page.screenshot(path=str(png_path))
        browser.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trip_map = build_map(args)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    trip_map.save(str(args.out_html))
    print(f"wrote {args.out_html} ({args.out_html.stat().st_size / 1024:,.0f} kB)")
    if args.out_png is not None:
        screenshot(args.out_html, args.out_png)
        print(f"wrote {args.out_png} ({args.out_png.stat().st_size / 1024:,.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
