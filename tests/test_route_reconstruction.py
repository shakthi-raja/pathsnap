"""Synthetic coverage of directed reconstruction and the API map geometry."""

from types import SimpleNamespace

import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import LineString

from src.api.service import MAX_POLYLINE_JOIN_M, _polylines
from src.routing.reconstruction import reconstruct_route
from src.routing.transitions import NetworkTravel


@pytest.fixture
def network():
    nodes = [
        (0, 0), (20, 0), (20, 20), (40, 20), (40, 40),
        (200, 0), (220, 0), (200, 100), (220, 100),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 0), (5, 6), (4, 5), (1, 2), (7, 8)]
    arcs = pd.DataFrame(
        edges,
        columns=["from_node_id", "to_node_id"],
    )
    geometries = [LineString([nodes[source], nodes[target]]) for source, target in edges]
    arcs["arc_id"] = arcs.index
    arcs["length_m"] = [geometry.length for geometry in geometries]
    weights = dict(zip(edges, arcs["length_m"], strict=True))
    sources, targets = zip(*weights)
    adjacency = csr_matrix((list(weights.values()), (sources, targets)), shape=(len(nodes),) * 2)
    # The synthetic graph is already in metres. Use a simple inverse for display.
    return SimpleNamespace(
        arcs=arcs,
        geometries=geometries,
        node_ids=range(len(nodes)),
        adjacency=adjacency,
        inverse=SimpleNamespace(transform=lambda x, y: (x / 100_000, y / 100_000)),
    )


def selected(network, *matches):
    rows = network.arcs.loc[[arc_id for arc_id, _ in matches]].copy()
    rows["position_m"] = rows["length_m"] * [fraction for _, fraction in matches]
    return rows


def traversed_arcs(pieces):
    return [[part.arc_id for part in piece] for piece in pieces]


def test_connector_geometry_and_fractional_endpoints(network):
    decoded = selected(network, (0, 0.25), (2, 0.75))
    travel = NetworkTravel(network.adjacency)
    lines = _polylines(decoded, network, travel)

    assert len(lines) == 1
    assert len(lines[0]) == 4
    for actual, expected in zip(
        lines[0],
        [(0, 0.00005), (0, 0.0002), (0.0002, 0.0002), (0.0002, 0.00035)],
        strict=True,
    ):
        assert actual == pytest.approx(expected, abs=1e-9)


def test_one_way_connector_cannot_be_used_in_reverse(network):
    decoded = selected(network, (3, 0.25), (3, 0.75), (2, 0.25), (2, 0.75))
    travel = NetworkTravel(network.adjacency)
    pieces = reconstruct_route(decoded, network, travel, max_join_m=MAX_POLYLINE_JOIN_M)

    assert traversed_arcs(pieces) == [[3], [2]]


def test_repeated_arc_merges_forward_progress_and_keeps_directed_revisit(network):
    decoded = selected(network, (0, 0.25), (0, 0.25), (0, 0.75), (0, 0.5))
    travel = NetworkTravel(network.adjacency)
    pieces = reconstruct_route(decoded, network, travel, max_join_m=MAX_POLYLINE_JOIN_M)

    assert traversed_arcs(pieces) == [[0, 1, 4, 0]]
    length = network.arcs.loc[0, "length_m"]
    assert pieces[0][0].start_m == pytest.approx(length * 0.25)
    assert pieces[0][0].end_m == pytest.approx(length)
    assert pieces[0][-1].end_m == pytest.approx(length * 0.5)


def test_disconnected_arcs_render_as_separate_polylines(network):
    decoded = selected(network, (0, 0.25), (0, 0.75), (8, 0.25), (8, 0.75))
    travel = NetworkTravel(network.adjacency)

    assert [len(line) for line in _polylines(decoded, network, travel)] == [2, 2]


def test_service_splits_a_reachable_connector_over_120_m(network):
    decoded = selected(network, (3, 0.25), (3, 0.75), (5, 0.25), (5, 0.75))
    travel = NetworkTravel(network.adjacency)
    assert MAX_POLYLINE_JOIN_M < travel.distance(4, 5) < 400

    assert [len(line) for line in _polylines(decoded, network, travel)] == [2, 2]


def test_connector_distance_limit_preserves_split(network):
    decoded = selected(network, (0, 0.25), (0, 0.75), (2, 0.25), (2, 0.75))
    travel = NetworkTravel(network.adjacency)
    distance = network.arcs.loc[1, "length_m"]

    assert traversed_arcs(reconstruct_route(decoded, network, travel, max_join_m=distance)) == [
        [0, 1, 2]
    ]
    assert traversed_arcs(
        reconstruct_route(decoded, network, travel, max_join_m=distance - 0.1)
    ) == [[0], [2]]
