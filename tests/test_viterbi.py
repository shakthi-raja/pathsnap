"""Tests for the Viterbi recurrence and emission normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from src.routing.transitions import emission_log_probs
from src.routing.viterbi import viterbi_decode


def test_connected_route_beats_the_pointwise_favourite() -> None:
    """The second observation prefers its own candidate, but the pair only works
    as a route when both observations take the first one."""
    emissions = [np.log([0.75, 0.25]), np.log([0.35, 0.65])]
    transitions = [np.log([[0.95, 0.05], [0.05, 0.95]])]

    assert viterbi_decode(emissions, transitions) == [0, 0]


def test_clear_emissions_win_when_transitions_are_weak() -> None:
    """With little connectivity preference, the confident observation wins."""
    emissions = [np.log([0.99, 0.01]), np.log([0.01, 0.99])]
    transitions = [np.log([[0.6, 0.4], [0.4, 0.6]])]

    assert viterbi_decode(emissions, transitions) == [0, 1]


def test_empty_input_returns_empty_path() -> None:
    assert viterbi_decode([], []) == []


def test_transition_count_must_match_observations() -> None:
    with pytest.raises(ValueError, match="one transition matrix"):
        viterbi_decode([np.log([0.5, 0.5]), np.log([0.5, 0.5])], [])


def test_observation_without_candidates_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        viterbi_decode([np.array([])], [])


def test_impossible_path_raises() -> None:
    emissions = [np.log([0.5, 0.5]), np.log([0.5, 0.5])]
    transitions = [np.full((2, 2), -np.inf)]

    with pytest.raises(ValueError, match="No feasible path"):
        viterbi_decode(emissions, transitions)


def test_transition_shape_is_validated() -> None:
    emissions = [np.log([0.5, 0.5]), np.log([0.5, 0.5])]
    transitions = [np.zeros((3, 2))]

    with pytest.raises(ValueError, match="has shape"):
        viterbi_decode(emissions, transitions)


def test_three_observation_path_maximises_the_joint_score() -> None:
    """Continuity beats the mild preference the last observation has."""
    emissions = [np.log([0.6, 0.4]), np.log([0.6, 0.4]), np.log([0.4, 0.6])]
    transitions = [
        np.log([[0.8, 0.2], [0.2, 0.8]]),
        np.log([[0.8, 0.2], [0.2, 0.8]]),
    ]

    assert viterbi_decode(emissions, transitions) == [0, 0, 0]


def test_emission_log_probs_normalise_to_a_distribution() -> None:
    log_probs = emission_log_probs(np.array([0.3, 0.6, 0.1]))

    assert np.isclose(np.exp(log_probs).sum(), 1.0)


def test_emission_floor_prevents_log_zero() -> None:
    log_probs = emission_log_probs(np.array([0.0, 1.0]))

    assert np.all(np.isfinite(log_probs))
