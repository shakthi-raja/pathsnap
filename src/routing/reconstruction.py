"""Recover continuous directed traversals from the decoder's selected arcs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from .transitions import NetworkTravel

if TYPE_CHECKING:
    from src.data.network import RoadNetwork


@dataclass(frozen=True, slots=True)
class ArcTraversal:
    arc_id: int
    start_m: float
    end_m: float


def reconstruct_route(
    decoded: pd.DataFrame,
    network: RoadNetwork,
    travel: NetworkTravel,
    *,
    max_join_m: float,
) -> list[list[ArcTraversal]]:
    """Return traveled arc portions, splitting at unavailable or long connectors."""
    if len(decoded) < 2:
        return []

    node_positions = {node_id: i for i, node_id in enumerate(network.node_ids)}
    connectors: dict[tuple[int, int], tuple[float, int]] = {}
    for arc in network.arcs.itertuples():
        nodes = (node_positions[arc.from_node_id], node_positions[arc.to_node_id])
        edge = (arc.length_m, int(arc.arc_id))
        # Match the sparse graph's minimum length, breaking parallel ties by arc ID.
        if nodes not in connectors or edge < connectors[nodes]:
            connectors[nodes] = edge

    selections = [
        (row, min(max(float(row.position_m), 0.0), float(row.length_m)))
        for row in decoded.itertuples()
    ]
    pieces: list[list[ArcTraversal]] = []
    current: list[ArcTraversal] = []
    for (previous, start), (following, end) in zip(selections, selections[1:]):
        if previous.arc_id == following.arc_id and start <= end:
            leg = [ArcTraversal(previous.arc_id, start, end)]
        else:
            path = travel.shortest_path(
                node_positions[previous.to_node_id],
                node_positions[following.from_node_id],
                max_distance_m=max_join_m,
            )
            if path is None:
                if current:
                    pieces.append(current)
                    current = []
                continue

            leg = [ArcTraversal(previous.arc_id, start, previous.length_m)]
            for source, target in zip(path, path[1:]):
                length, arc_id = connectors[source, target]
                leg.append(ArcTraversal(arc_id, 0.0, length))
            leg.append(ArcTraversal(following.arc_id, 0.0, end))

        for part in leg:
            if part.start_m == part.end_m:
                continue
            if current and current[-1].arc_id == part.arc_id and current[-1].end_m == part.start_m:
                part = ArcTraversal(part.arc_id, current.pop().start_m, part.end_m)
            current.append(part)

    if current:
        pieces.append(current)
    return pieces
