"""Server-side scoring of a completed prediction.

Ground truth is loaded here and nowhere else, and only after the predicted
route already exists. Nothing in this module can influence a prediction, which
is the property the guidelines care about most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["evaluate_prediction", "route_continuity"]


def route_continuity(decoded: pd.DataFrame, travel) -> float | None:
    """Fraction of consecutive decoded arcs that the road network can connect.

    Immediate node adjacency is too strict: two observations 1 s apart are
    often separated by several short arcs that never appear as candidates. The
    rule here is that a directed path must exist within the decoder's link
    limit, which is what the guideline asks for.
    """
    if len(decoded) < 2:
        return None
    rows = list(decoded.itertuples())
    connected = 0
    for previous, current in zip(rows[:-1], rows[1:], strict=True):
        if int(previous.arc_id) == int(current.arc_id):
            connected += 1
            continue
        distance = travel.distance(int(previous.to_node_id), int(current.from_node_id))
        if distance is not None:
            connected += 1
    return connected / (len(rows) - 1)


def evaluate_prediction(
    decoded: pd.DataFrame,
    classifier_only: pd.DataFrame,
    alignment: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    travel,
    classifier_route: pd.DataFrame | None = None,
) -> dict[str, float | int | None]:
    """Score a prediction against the aligned reference route.

    Args:
        decoded: One row per observation holding the Viterbi ``arc_id``.
        classifier_only: One row per observation with the highest scoring
            candidate, used for the before/after comparison.
        alignment: Output of :func:`src.data.labels.align_reference_route`.
        candidate_pairs: Candidate rows with ``gps_index`` and ``arc_id``.

    Returns:
        Metric values, with ``None`` where a metric is not applicable.
    """
    confident = alignment.loc[alignment["alignment_confident"], ["gps_index", "aligned_arc_id"]]
    if confident.empty:
        return {
            "pointwise_accuracy": None,
            "top1_accuracy": None,
            "route_continuity": route_continuity(decoded, travel),
            "classifier_continuity": (
                route_continuity(classifier_route, travel) if classifier_route is not None else None
            ),
            "candidate_miss_rate": None,
            "evaluated_observations": 0,
        }

    scored = confident.merge(
        decoded[["gps_index", "arc_id"]].rename(columns={"arc_id": "decoded_arc_id"}),
        on="gps_index",
        how="inner",
    )
    scored = scored.merge(
        classifier_only[["gps_index", "arc_id"]].rename(columns={"arc_id": "top1_arc_id"}),
        on="gps_index",
        how="inner",
    )

    decoded_correct = float((scored["decoded_arc_id"] == scored["aligned_arc_id"]).mean())
    top1_correct = float((scored["top1_arc_id"] == scored["aligned_arc_id"]).mean())

    candidates_per_observation = candidate_pairs.groupby("gps_index")["arc_id"].apply(set)
    misses = [
        row.aligned_arc_id not in candidates_per_observation.get(row.gps_index, set())
        for row in scored.itertuples()
    ]

    return {
        "pointwise_accuracy": decoded_correct,
        "top1_accuracy": top1_correct,
        "route_continuity": route_continuity(decoded, travel),
        "classifier_continuity": (
            route_continuity(classifier_route, travel) if classifier_route is not None else None
        ),
        "candidate_miss_rate": float(np.mean(misses)),
        "evaluated_observations": int(len(scored)),
    }
