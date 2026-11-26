"""Local HTTP endpoints for a saved packet CNN."""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pickle import UnpicklingError
from typing import Annotated, Literal

from clickhouse_connect.driver.exceptions import ClickHouseError
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from traffic_classifier.clickhouse_client import connect
from traffic_classifier.inference import PacketInput, Prediction, Predictor, load_predictor
from traffic_classifier.ml.dataset import MAX_PACKETS
from traffic_classifier.ml.packets import InvalidPackets
from traffic_classifier.prediction_statistics import (
    CompletedRun,
    PredictionStatistics,
    RunNotFound,
    completed_runs,
    prediction_statistics,
)
from traffic_classifier.vnat_groups import PREFIX_LENGTHS

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    offsets_us: list[int] = Field(min_length=1, max_length=MAX_PACKETS, examples=[[0, 2000, 5250]])
    sizes_bytes: list[int] = Field(min_length=1, max_length=MAX_PACKETS, examples=[[100, 300, 180]])
    directions: list[int] = Field(min_length=1, max_length=MAX_PACKETS, examples=[[1, 0, 1]])


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    model_id: str
    n: int


def require_model(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(503, "Model is unavailable. Prepare the saved CNN bundle and restart.")
    return predictor


def create_app(loader: Callable[[], Predictor] = load_predictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = loader()
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            EOFError,
            UnpicklingError,
        ) as error:
            logger.warning("Could not load the local CNN bundle: %s", error)
        yield
        app.state.predictor = None

    app = FastAPI(title="TrafficClassifier", version="1.0", lifespan=lifespan)
    app.state.predictor = None

    @app.get("/", include_in_schema=False)
    def open_swagger() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/ready")
    def ready(request: Request) -> ReadyResponse:
        predictor = require_model(request)
        return ReadyResponse(status="ready", model_id=predictor.model_id, n=predictor.n)

    @app.post(
        "/api/v1/predict",
        responses={
            422: {"description": "Invalid packet input"},
            503: {"description": "Model unavailable"},
        },
    )
    def predict(payload: PredictRequest, request: Request) -> Prediction:
        predictor = require_model(request)
        try:
            return predictor.predict(
                PacketInput(payload.offsets_us, payload.sizes_bytes, payload.directions)
            )
        except InvalidPackets as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/v1/runs")
    def list_runs() -> list[CompletedRun]:
        try:
            with connect() as client:
                return completed_runs(client)
        except (ClickHouseError, OSError, RuntimeError, ValueError) as error:
            logger.warning("Could not list completed prediction runs: %s", error)
            raise HTTPException(503, "ClickHouse prediction runs are unavailable.") from error

    @app.get("/api/v1/statistics")
    def statistics(
        run_id: Annotated[str | None, Query(pattern="^[0-9a-f]{64}$")] = None,
        model_id: str | None = None,
        n: int | None = None,
    ) -> PredictionStatistics:
        if n is not None and n not in PREFIX_LENGTHS:
            raise HTTPException(422, f"Expected packet count in {PREFIX_LENGTHS}.")
        try:
            with connect() as client:
                return prediction_statistics(client, run_id=run_id, model_id=model_id, n=n)
        except RunNotFound as error:
            raise HTTPException(404, str(error)) from error
        except (ClickHouseError, OSError, RuntimeError, ValueError) as error:
            logger.warning("Could not read prediction statistics: %s", error)
            raise HTTPException(503, "ClickHouse statistics are unavailable.") from error

    return app


app = create_app()
