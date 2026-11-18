"""Viterbi decoding over candidate sequences.

The dynamic program is deliberately free of any dependency on the road network
or the dataset: it takes log emission scores per observation and log transition
scores between consecutive observations, and returns the most probable index
per observation. That makes it possible to test the recurrence on a handful of
numbers rather than on a whole trip.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["viterbi_decode"]

NEGATIVE_INFINITY = -np.inf


def viterbi_decode(
    log_emissions: Sequence[np.ndarray],
    log_transitions: Sequence[np.ndarray],
) -> list[int]:
    """Return the highest scoring path through the candidate grid.

    Args:
        log_emissions: One array per observation, holding the log emission for
            each candidate of that observation.
        log_transitions: One ``(n_previous, n_current)`` matrix per consecutive
            pair of observations.

    Returns:
        One candidate index per observation.

    Raises:
        ValueError: If the input lengths are inconsistent, an observation has no
            candidates, or no path is possible.
    """
    if not log_emissions:
        return []
    if len(log_transitions) != len(log_emissions) - 1:
        raise ValueError("Expected one transition matrix per consecutive observation pair")
    if any(len(emission) == 0 for emission in log_emissions):
        raise ValueError("Every observation needs at least one candidate")

    scores = np.asarray(log_emissions[0], dtype=float).copy()
    if not np.isfinite(scores).any():
        raise ValueError("No feasible candidate at observation 0")

    backpointers: list[np.ndarray] = []
    for index in range(1, len(log_emissions)):
        transition = np.asarray(log_transitions[index - 1], dtype=float)
        emission = np.asarray(log_emissions[index], dtype=float)
        if transition.shape != (len(scores), len(emission)):
            raise ValueError(
                f"Transition {index - 1} has shape {transition.shape}, "
                f"expected {(len(scores), len(emission))}"
            )

        proposed = scores[:, None] + transition
        best_previous = np.argmax(proposed, axis=0)
        scores = proposed[best_previous, np.arange(len(best_previous))] + emission

        if not np.isfinite(scores).any():
            raise ValueError(f"No feasible path at observation {index}")
        backpointers.append(best_previous)

    chosen = [int(np.argmax(scores))]
    for previous in reversed(backpointers):
        chosen.append(int(previous[chosen[-1]]))
    chosen.reverse()
    return chosen
