"""Pydantic schemas for the fraud detection API."""

from math import isfinite
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_FEATURE_COUNT = 256
"""Maximum number of features accepted in one prediction request."""


class PredictionRequest(BaseModel):
    """A single transaction represented by finite numeric feature values."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Annotated[float, Field(description="Finite feature value")]]

    @field_validator("features", mode="before")
    @classmethod
    def validate_feature_types(cls, features: object) -> object:
        if isinstance(features, dict):
            for name, value in features.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(  # noqa: TRY004
                        f"features[{name!r}] must be a real numeric value"
                    )
        return features

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: dict[str, float]) -> dict[str, float]:
        if len(features) > MAX_FEATURE_COUNT:
            raise ValueError(
                f"features must contain at most {MAX_FEATURE_COUNT} entries"
            )
        for name, value in features.items():
            if not isfinite(value):
                raise ValueError(
                    f"features[{name!r}] must be a finite number (not NaN or infinity)"
                )
        return features


class PredictionResponse(BaseModel):
    """Prediction result returned by the inference endpoint."""

    is_fraud: bool
    fraud_probability: float
    threshold: float
    model_version: str


class ModelInfoResponse(BaseModel):
    """Trusted model metadata exposed by the service.

    Legacy schema 1.0 artifacts may not contain split or dataset metadata. Those
    fields are optional and are returned as ``None`` rather than failing startup.
    """

    model_name: str | None = None
    model_version: str | None = None
    model_identity: dict[str, str] | None = None
    feature_names: list[str] = Field(default_factory=list)
    threshold: float
    artifact_schema_version: str | None = None
    split_strategy: str | None = None
    dataset_sha256: str | None = None
    created_at_utc: str | None = None


class HealthResponse(BaseModel):
    """Service liveness response."""

    status: str
    service: str
