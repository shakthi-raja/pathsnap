"""Synthetic Parquet checks for the evaluator path and its API contract."""

import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.main import create_app
from src.api.schemas import PredictRequest
from src.data.evaluator import SCORE_COLUMNS, EvaluatorInputError, load_evaluator_scores
from src.visualization.maps import normalise_payload


@pytest.fixture
def evaluator_inputs(tmp_path):
    root = tmp_path / "raw"
    trip = root / "00000001"
    trip.mkdir(parents=True)
    files = {
        "track": "0.00005 0 0\n0.0002 0.00005 1\n0.00035 0.0002 2\n",
        "nodes": "0 0\n0.0002 0\n0.0002 0.0002\n0.0004 0.0002\n",
        "arcs": "0 1\n1 2\n2 3\n",
        "route": "Reference data must never be parsed in evaluator mode.\n",
    }
    for extension, content in files.items():
        (trip / f"00000001.{extension}").write_text(content, encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    return root, predictions


def test_evaluator_prediction_uses_supplied_scores_and_recovers_connector(
    evaluator_inputs, monkeypatch
):
    root, predictions = evaluator_inputs
    pd.DataFrame([(1, 2, 2, 1.0), (1, 0, 0, 1.0), (1, 2, 0, 0.0)], columns=SCORE_COLUMNS).to_parquet(
        predictions / "part-a.parquet", index=False
    )
    monkeypatch.setenv("PATHSNAP_EVALUATOR_PREDICTIONS", str(predictions))
    monkeypatch.setenv("PATHSNAP_EVALUATION_ROOT", str(root))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Evaluator mode must use only the supplied candidate set and probabilities")

    monkeypatch.setattr("src.api.service.joblib.load", unexpected_call)
    monkeypatch.setattr("src.api.service.generate_candidates", unexpected_call)
    monkeypatch.setattr("src.api.service.align_reference_route", unexpected_call)
    endpoints = {route.path: route.endpoint for route in create_app().routes}

    assert endpoints["/health"]()["model"] == "Precomputed evaluator scores"
    assert endpoints["/trips"]().trip_ids == ["00000001"]
    predict = endpoints["/predict"]
    result = predict(PredictRequest(trip_id="1"))
    assert result.status == "ok"
    assert result.metrics == {"route_continuity": 1.0}
    assert result.warnings == ["1 observations had no scored candidates."]
    assert len(result.predicted_polylines) == 1
    payload = normalise_payload(result.model_dump())
    expected = [(0, 0.00005), (0, 0.0002), (0.0002, 0.0002), (0.0002, 0.00035)]
    for actual, point in zip(payload.predicted_polylines[0], expected, strict=True):
        assert actual == pytest.approx(point, abs=1e-9)

    with pytest.raises(HTTPException) as unscored:
        predict(PredictRequest(trip_id="2"))
    assert unscored.value.status_code == 404


@pytest.mark.parametrize("key", ["gps_index", "arc_id"])
def test_evaluator_rejects_keys_missing_from_raw_trip(evaluator_inputs, monkeypatch, key):
    root, predictions = evaluator_inputs
    scores = pd.DataFrame([("00000001", 0, 0, 1.0)], columns=SCORE_COLUMNS)
    scores[key] = 999
    scores.to_parquet(predictions / "part-a.parquet", index=False)
    monkeypatch.setenv("PATHSNAP_EVALUATOR_PREDICTIONS", str(predictions))
    monkeypatch.setenv("PATHSNAP_EVALUATION_ROOT", str(root))

    predict = next(route.endpoint for route in create_app().routes if route.path == "/predict")
    with pytest.raises(HTTPException, match=key) as bad_key:
        predict(PredictRequest(trip_id="1"))
    assert bad_key.value.status_code == 422


def test_load_evaluator_scores_groups_trips_across_shards(tmp_path):
    pd.DataFrame([(2, 0, 1, 0.4), (1, 2, 3, 0.2)], columns=SCORE_COLUMNS).to_parquet(
        tmp_path / "part-a.parquet", index=False
    )
    pd.DataFrame([("00000001", 0, 4, 0.8)], columns=SCORE_COLUMNS).to_parquet(
        tmp_path / "part-b.parquet", index=False
    )
    grouped = load_evaluator_scores(tmp_path)
    assert list(grouped) == ["00000001", "00000002"]
    assert grouped["00000001"]["gps_index"].tolist() == [0, 2]
    assert grouped["00000001"]["p_correct"].tolist() == [0.8, 0.2]

    pd.DataFrame([("1", 2, 3, 0.9)], columns=SCORE_COLUMNS).to_parquet(
        tmp_path / "part-c.parquet", index=False
    )
    with pytest.raises(EvaluatorInputError, match="Duplicate evaluator candidate keys"):
        load_evaluator_scores(tmp_path)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_evaluator_probability_is_rejected(tmp_path, invalid):
    pd.DataFrame([(1, 0, 10, invalid)], columns=SCORE_COLUMNS).to_parquet(
        tmp_path / "part-a.parquet", index=False
    )
    with pytest.raises(EvaluatorInputError, match="finite probabilities"):
        load_evaluator_scores(tmp_path)
