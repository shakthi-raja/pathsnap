"""Route reconstruction: classifier scores in, one arc per observation out.

The decoder is the step that turns independent per-candidate scores into a
drivable path. It groups candidates by observation, converts scores into
emissions, scores transitions against the road network and the elapsed time,
and runs Viterbi over the resulting grid.

The classifier-only path is kept alongside the decoded one so the two can be
compared on identical data, which is what the report has to show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transitions import (
    CandidateState,
    NetworkTravel,
    emission_log_probs,
    transition_log_scores,
)
from .viterbi import viterbi_decode

__all__ = ["decode_candidates", "DecodedRoute"]

#: Feature columns that hold the state geometry, not model inputs.
_STATE_COLUMNS = (
    "gps_index",
    "arc_id",
    "from_node_id",
    "to_node_id",
    "length_m",
    "position_m",
    "projected_x",
    "projected_y",
    "movement_m",
    "dt_s",
)


@dataclass(frozen=True, slots=True)
class DecodedRoute:
    """Decoded arcs plus the classifier-only baseline for the same trip.

    Attributes:
        decoded: One row per observation with the Viterbi choice.
        classifier_only: One row per observation with the highest scoring
            candidate, ignoring connectivity. Used for the before/after test.
    """

    decoded: pd.DataFrame
    classifier_only: pd.DataFrame


def _state(row: pd.Series) -> CandidateState:
    return CandidateState(
        arc_id=int(row["arc_id"]),
        from_node=int(row["from_node_id"]),
        to_node=int(row["to_node_id"]),
        length_m=float(row["length_m"]),
        position_m=float(row["position_m"]),
        x=float(row["projected_x"]),
        y=float(row["projected_y"]),
    )


def decode_candidates(
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    travel: NetworkTravel,
    max_speed_mps: float = 50.0,
) -> DecodedRoute:
    """Decode one trip.

    Args:
        candidates: Candidate rows for the trip, ordered by observation. Must
            contain the geometry columns listed in ``_STATE_COLUMNS``.
        probabilities: Positive-class probability for each candidate row, in the
            same order as ``candidates``.
        travel: Bounded network distance helper for this trip's road graph.
        max_speed_mps: Speed ceiling used to reject impossible transitions.

    Returns:
        A :class:`DecodedRoute` with the decoded and classifier-only choices.

    Raises:
        ValueError: If the candidate table is empty or a required column is
            missing.
    """
    missing = [column for column in _STATE_COLUMNS if column not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")
    if candidates.empty:
        raise ValueError("Cannot decode an empty candidate table")
    if len(candidates) != len(probabilities):
        raise ValueError("Expected one probability per candidate row")

    scored = candidates.loc[:, list(_STATE_COLUMNS)].copy()
    scored["p_positive"] = np.asarray(probabilities, dtype=float)

    decoded_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    observation_ids = scored["gps_index"].to_numpy()
    boundaries = np.flatnonzero(np.diff(observation_ids)) + 1
    groups = np.split(np.arange(len(scored)), boundaries)

    previous_states: list[CandidateState] | None = None
    transition_matrices: list[np.ndarray] = []
    emissions: list[np.ndarray] = []
    group_slices: list[np.ndarray] = []

    for positions in groups:
        block = scored.iloc[positions]
        states = [_state(row) for _, row in block.iterrows()]
        emissions.append(emission_log_probs(block["p_positive"].to_numpy()))
        group_slices.append(positions)

        if previous_states is not None:
            first = block.iloc[0]
            transition_matrices.append(
                transition_log_scores(
                    previous_states,
                    states,
                    gps_m=float(first["movement_m"]) if np.isfinite(first["movement_m"]) else 0.0,
                    dt_s=float(first["dt_s"]) if np.isfinite(first["dt_s"]) else 0.0,
                    travel=travel,
                    max_speed_mps=max_speed_mps,
                )
            )
        previous_states = states

    try:
        chosen = viterbi_decode(emissions, transition_matrices)
    except ValueError:
        chosen = [int(np.argmax(emission)) for emission in emissions]

    for positions, index, emission in zip(group_slices, chosen, emissions, strict=True):
        block = scored.iloc[positions]
        best = block.iloc[index]
        classifier_best = block.iloc[int(np.argmax(emission))]

        decoded_rows.append(
            {
                "gps_index": int(best["gps_index"]),
                "arc_id": int(best["arc_id"]),
                "p_positive": float(best["p_positive"]),
                "from_node_id": int(best["from_node_id"]),
                "to_node_id": int(best["to_node_id"]),
                "length_m": float(best["length_m"]),
                "position_m": float(best["position_m"]),
                "projected_x": float(best["projected_x"]),
                "projected_y": float(best["projected_y"]),
                "candidate_count": int(len(block)),
            }
        )
        baseline_rows.append(
            {
                "gps_index": int(classifier_best["gps_index"]),
                "arc_id": int(classifier_best["arc_id"]),
                "p_positive": float(classifier_best["p_positive"]),
            }
        )

    return DecodedRoute(
        decoded=pd.DataFrame(decoded_rows),
        classifier_only=pd.DataFrame(baseline_rows),
    )
