"""Transition scoring for the route decoder.

A classifier scores each candidate in isolation, so consecutive observations can
land on arcs that no vehicle could drive between. This module measures how far
you would actually have to drive on the directed road network to get from one
candidate to the next, and scores that against how far the GPS moved in the
elapsed time.

Distances come from a bounded Dijkstra over the sparse adjacency matrix. The
search stops at a link limit, so a single query only touches the handful of
nodes reachable within that radius rather than the whole city.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

__all__ = ["NetworkTravel", "CandidateState", "transition_log_scores", "emission_log_probs"]

#: Never consider a road detour longer than this when linking two candidates.
DEFAULT_MAX_LINK_M = 400.0
#: Effective speed ceiling, in metres per second (about 180 km/h).
DEFAULT_MAX_SPEED_MPS = 50.0
#: Penalty applied when the route implies driving backwards, in log space.
BACKWARDS_PENALTY = -6.0


@dataclass(frozen=True, slots=True)
class CandidateState:
    """The geometry of one candidate that the decoder needs."""

    arc_id: int
    from_node: int
    to_node: int
    length_m: float
    position_m: float
    x: float
    y: float


class NetworkTravel:
    """Bounded road-network distances between node positions.

    Node identifiers in this dataset are row positions into the node table, so
    they index the adjacency matrix directly.
    """

    def __init__(self, adjacency, *, max_link_m: float = DEFAULT_MAX_LINK_M) -> None:
        self._adjacency = adjacency
        self._indptr = adjacency.indptr
        self._indices = adjacency.indices
        self._data = adjacency.data
        self._max_link_m = max_link_m
        self._reachable: dict[int, dict[int, float]] = {}
        self._predecessors: dict[int, dict[int, int]] = {}

    def _expand(self, source: int) -> dict[int, float]:
        """Distances from ``source`` to every node within the link limit."""
        cached = self._reachable.get(source)
        if cached is not None:
            return cached

        limit = self._max_link_m
        distances = {source: 0.0}
        predecessors: dict[int, int] = {}
        heap = [(0.0, source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance > distances.get(node, np.inf):
                continue
            if distance > limit:
                break
            for index in range(self._indptr[node], self._indptr[node + 1]):
                target = int(self._indices[index])
                candidate = distance + float(self._data[index])
                if candidate <= limit and candidate < distances.get(target, np.inf):
                    distances[target] = candidate
                    predecessors[target] = node
                    heapq.heappush(heap, (candidate, target))

        self._predecessors[source] = predecessors
        self._reachable[source] = distances
        return distances

    def distance(self, source: int, target: int) -> float | None:
        """Directed road distance, or ``None`` when it exceeds the link limit."""
        if source == target:
            return 0.0
        distance = self._expand(source).get(target)
        return None if distance is None else float(distance)

    def shortest_path(
        self, source: int, target: int, *, max_distance_m: float
    ) -> list[int] | None:
        """Return the directed connector's node positions within the given limit."""
        distance = self.distance(source, target)
        if distance is None or distance > max_distance_m:
            return None

        path = [target]
        while path[-1] != source:
            path.append(self._predecessors[source][path[-1]])
        path.reverse()
        return path


def _travel_distance(
    previous: CandidateState, current: CandidateState, travel: NetworkTravel
) -> float | None:
    """Road distance to move from one candidate to the next."""
    if previous.arc_id == current.arc_id:
        progress = current.position_m - previous.position_m
        return max(progress, 0.0)

    remainder = max(previous.length_m - previous.position_m, 0.0)
    link = travel.distance(previous.to_node, current.from_node)
    if link is None:
        return None
    return remainder + link + current.position_m


def transition_log_scores(
    previous: list[CandidateState],
    current: list[CandidateState],
    *,
    gps_m: float,
    dt_s: float,
    travel: NetworkTravel,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> np.ndarray:
    """Log transition scores for one consecutive pair of observations.

    The score rewards road distance that matches the GPS displacement and
    punishes detours, backwards motion and speeds no vehicle could reach.
    """
    scores = np.full((len(previous), len(current)), -np.inf)
    tolerance = max(25.0, 0.5 * gps_m)
    gps_m = max(gps_m, 0.0)

    for i, previous_state in enumerate(previous):
        for j, current_state in enumerate(current):
            gap = float(
                np.hypot(current_state.x - previous_state.x, current_state.y - previous_state.y)
            )
            road_m = _travel_distance(previous_state, current_state, travel)

            if road_m is None:
                # No road link within the search limit. Keep the pair feasible
                # but heavily penalise it, scaled by the straight-line gap, so
                # one unreachable pair cannot make the whole trip undecodable.
                if gap > 2000.0:
                    continue
                scores[i, j] = -30.0 - gap / 50.0
                continue

            mismatch = abs(road_m - gps_m)
            score = -mismatch / tolerance

            if (
                previous_state.arc_id == current_state.arc_id
                and current_state.position_m < previous_state.position_m
            ):
                score += BACKWARDS_PENALTY

            if dt_s > 0 and road_m / dt_s > max_speed_mps:
                score -= 12.0

            scores[i, j] = score

    return scores


def emission_log_probs(scores: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    """Turn per-candidate probabilities into a normalised log distribution.

    A binary positive-class probability is not a distribution across the
    candidates of one observation, so it has to be normalised before the
    decoder can compare states.
    """
    weights = np.clip(np.asarray(scores, dtype=float), floor, None)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(weights), -np.log(len(weights)))
    return np.log(weights / total)
