"""FastAPI backend for PathSnap.

Laptop 1 runs this. The interface on Laptop 2 sends a trip ID and receives the
reconstructed route plus the server-side score. Reference routes never leave
this process.

Run with::

    python -m src.api.main
    # or
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from src.data.evaluator import EvaluatorInputError
from src.data.reader import normalize_trip_id

from .schemas import PredictRequest, PredictResponse, TripList
from .service import PathSnapService

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("pathsnap.api")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"


def _default_evaluation_root() -> Path:
    """Trip folders for the server to serve.

    Deployments supply this explicitly. Locally we fall back to the student-test
    partition so the system can be exercised before the evaluator partition
    arrives.
    """
    configured = os.environ.get("PATHSNAP_EVALUATION_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    data_root = os.environ.get("PATHSNAP_DATA_ROOT")
    if data_root:
        return (Path(data_root).expanduser() / "student_test").resolve()
    return (PROJECT_ROOT.parent / "data" / "raw" / "student_test").resolve()


def create_app(service: PathSnapService | None = None) -> FastAPI:
    """Build the application, optionally with an injected service.

    Injection keeps the endpoints testable without a trained model on disk.
    """
    app = FastAPI(
        title="PathSnap API",
        description="Distributed map-matching and route reconstruction.",
        version="1.0.0",
    )

    state: dict[str, PathSnapService] = {}
    if service is not None:
        state["service"] = service

    def get_service() -> PathSnapService:
        existing = state.get("service")
        if existing is not None:
            return existing
        evaluator_predictions = os.environ.get("PATHSNAP_EVALUATOR_PREDICTIONS")
        if evaluator_predictions and not os.environ.get("PATHSNAP_EVALUATION_ROOT"):
            raise HTTPException(
                status_code=503,
                detail="Set PATHSNAP_EVALUATION_ROOT to the raw evaluator trip directory.",
            )
        try:
            built = PathSnapService(
                evaluation_root=_default_evaluation_root(),
                model_path=os.environ.get(
                    "PATHSNAP_MODEL_PATH", str(DEFAULT_ARTIFACTS / "final_pipeline.joblib")
                ),
                metadata_path=os.environ.get(
                    "PATHSNAP_MODEL_METADATA", str(DEFAULT_ARTIFACTS / "model_metadata.json")
                ),
                evaluator_predictions=evaluator_predictions,
            )
        except (FileNotFoundError, EvaluatorInputError) as exc:
            logger.error("Prediction service unavailable: %s", exc)
            if evaluator_predictions:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            raise HTTPException(
                status_code=503, detail="Check the backend dataset root and model artifact paths."
            ) from exc
        state["service"] = built
        return built

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Print every request so the evaluator can see the pipeline run."""
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "%s %s -> %d in %.0f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health")
    def health() -> dict[str, object]:
        service_instance = get_service()
        return {
            "status": "ok",
            "trips": len(service_instance.list_trips()),
            "model": service_instance.metadata.get("model_name"),
        }

    @app.get("/trips", response_model=TripList)
    def trips() -> TripList:
        return TripList(trip_ids=get_service().list_trips())

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        try:
            trip_id = normalize_trip_id(payload.trip_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        service_instance = get_service()
        try:
            result = service_instance.predict(trip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EvaluatorInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(
            trip_id=result.trip_id,
            status=result.status,
            track_points=result.track_points,
            road_context=result.road_context,
            predicted_polylines=result.predicted_polylines,
            metrics=dict(result.metrics),
            runtime_ms=result.runtime_ms,
            warnings=list(result.warnings),
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("PATHSNAP_HOST", "0.0.0.0"),
        port=int(os.environ.get("PATHSNAP_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
