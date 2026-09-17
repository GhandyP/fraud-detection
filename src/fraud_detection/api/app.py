"""FastAPI application factory for trusted model inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from fraud_detection.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from fraud_detection.models.predict import ModelPredictor
from fraud_detection.utils.config import ProjectPaths, load_config
from fraud_detection.utils.logger import get_logger

SERVICE_NAME = "fraud-detection-api"
DEFAULT_MAX_BODY_BYTES = 32 * 1024
DEFAULT_CORS_ORIGINS = [
    "http://localhost:4321",
    "http://127.0.0.1:4321",
]
logger = get_logger(__name__)


def _max_body_bytes() -> int:
    raw = os.environ.get("FRAUD_API_MAX_BODY_BYTES")
    if raw is None:
        return DEFAULT_MAX_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES
    return value if value > 0 else DEFAULT_MAX_BODY_BYTES


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than the configured byte limit."""

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > self.max_body_bytes
            except ValueError:
                too_large = False
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            "Request body exceeds the maximum size of "
                            f"{self.max_body_bytes} bytes"
                        )
                    },
                )
        body = await request.body()
        if len(body) > self.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        "Request body exceeds the maximum size of "
                        f"{self.max_body_bytes} bytes"
                    )
                },
            )
        response = await call_next(request)
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add and log a stable request identifier for every HTTP response."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request method=%s path=%s status=%s request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
        )
        return response


def _model_info(predictor: ModelPredictor) -> ModelInfoResponse:
    metadata: dict[str, Any] = dict(predictor.metadata)
    identity = metadata.get("model_identity")
    return ModelInfoResponse(
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        model_identity=dict(identity) if isinstance(identity, dict) else None,
        feature_names=list(predictor.feature_names),
        threshold=predictor.threshold,
        artifact_schema_version=metadata.get("artifact_schema_version"),
        split_strategy=metadata.get("split_strategy"),
        dataset_sha256=metadata.get("dataset_sha256"),
        created_at_utc=metadata.get("created_at_utc"),
    )


def create_app(
    root: Path | None = None,
    config_path: str | Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create an API bound to the configured, trusted model artifact."""
    project_root = Path.cwd() if root is None else Path(root)
    model_name = "<configured-model>"
    try:
        config = load_config(project_root, config_path)
        model_name = config.train.model_name
        paths = ProjectPaths(project_root)
        artifact_path = paths.models_trained / f"{model_name}.joblib"
        predictor = ModelPredictor(artifact_path)
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize fraud-detection-api: "
            "load a valid configured model artifact at "
            f"<root>/models/trained/{model_name}.joblib ({exc})"
        ) from exc

    app = FastAPI(title=SERVICE_NAME)
    app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=_max_body_bytes())
    origins = (
        cors_origins
        if cors_origins is not None
        else [
            origin.strip()
            for origin in os.environ.get(
                "FRAUD_API_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)
            ).split(",")
            if origin.strip()
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    # Added last so the request identifier wraps every response, including 413.
    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": jsonable_encoder(exc.errors())}
        )

    @app.exception_handler(ValueError)
    async def predictor_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service=SERVICE_NAME)

    model_info = _model_info(predictor)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_information() -> ModelInfoResponse:
        return model_info

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        features = pd.DataFrame([request.features])
        prediction = predictor.predict(features)
        probabilities = predictor.predict_proba(features)
        probability = float(probabilities[0, 1])
        threshold = float(predictor.threshold)
        metadata = predictor.metadata
        return PredictionResponse(
            is_fraud=bool(prediction[0]),
            fraud_probability=probability,
            threshold=threshold,
            model_version=str(metadata.get("model_version", "")),
        )

    return app
