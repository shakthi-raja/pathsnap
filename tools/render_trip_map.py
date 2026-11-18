"""Render a Folium map from raw trip files, for EDA and the technical report.

Produces the geographical evidence the report requires: GPS observations,
road-network geometry and the reference route, on a **training or
student-test** trip, where the reference is permitted. The evaluator UI must
never use this tool.

Usage::

    python tools/render_trip_map.py --trip 00000062 --partition train \\
        --data-root ../data/raw --out results/maps/trip_00000062.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python tools/render_trip_map.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.maps import build_trip_map_from_files  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", required=True, help="Trip id, e.g. 00000062")
    parser.add_argument(
        "--partition", default="train", choices=("train", "student_test"), help="Dataset partition"
    )
    parser.add_argument("--data-root", type=Path, default=Path("../data/raw"))
    parser.add_argument("--out", type=Path, required=True, help="Destination .html file")
    parser.add_argument(
        "--max-road-segments",
        type=int,
        default=15_000,
        help="Cap on rendered road segments (browser performance).",
    )
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="Include the reference route. Only valid on train/student_test data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trip_dir = args.data_root / args.partition / args.trip
    if not trip_dir.is_dir():
        print(f"error: no such trip directory: {trip_dir}", file=sys.stderr)
        return 2

    stem = args.trip
    route_path = trip_dir / f"{stem}.route" if args.with_reference else None
    if args.with_reference and not route_path.is_file():
        print(f"error: reference route not found: {route_path}", file=sys.stderr)
        return 2

    trip_map = build_trip_map_from_files(
        track_path=str(trip_dir / f"{stem}.track"),
        nodes_path=str(trip_dir / f"{stem}.nodes"),
        arcs_path=str(trip_dir / f"{stem}.arcs"),
        route_path=str(route_path) if route_path else None,
        max_road_segments=args.max_road_segments,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    trip_map.save(str(args.out))
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:,.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
