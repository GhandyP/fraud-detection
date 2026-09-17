from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from fraud_detection.api import create_app
from fraud_detection.models.predict import ModelPredictor
from fraud_detection.models.train import ModelTrainer, TrainConfig


def _project(tmp_path: Path, *, artifact: bool = True) -> Path:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "models" / "trained").mkdir(parents=True)
    config = {
        "dataset": {"file_name": "synthetic.csv", "target_column": "target"},
        "train": {
            "model_name": "api_model",
            "model_version": "test-1",
            "threshold": 0.4,
            "split_strategy": "random",
        },
    }
    (root / "config" / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    if artifact:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(24, 2))
        target = (features[:, 0] + features[:, 1] > 0).astype(int)
        trainer = ModelTrainer(
            TrainConfig(
                model_dir=root / "models" / "trained",
                model_name="api_model",
                model_version="test-1",
                threshold=0.4,
                dataset_sha256="a" * 64,
            )
        )
        trainer.train_validate_test(
            features[:12],
            target[:12],
            features[12:18],
            target[12:18],
            features[18:],
            target[18:],
            feature_names=["amount", "balance"],
        )
    return root


def test_health_and_model_info(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with TestClient(create_app(root=root)) as client:
        health = client.get("/health")
        info = client.get("/model-info")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "fraud-detection-api"}
    assert info.status_code == 200
    body = info.json()
    assert body["model_name"] == "api_model"
    assert body["model_version"] == "test-1"
    assert body["model_identity"] == {"name": "api_model", "version": "test-1"}
    assert body["feature_names"] == ["amount", "balance"]
    artifact = ModelPredictor(root / "models" / "trained" / "api_model.joblib")
    assert body["threshold"] == artifact.threshold
    assert (
        body["artifact_schema_version"] == artifact.metadata["artifact_schema_version"]
    )
    assert body["split_strategy"] == artifact.metadata["split_strategy"]
    assert body["dataset_sha256"] == artifact.metadata["dataset_sha256"]
    assert body["created_at_utc"] == artifact.metadata["created_at_utc"]


def test_missing_artifact_fails_clearly(tmp_path: Path) -> None:
    root = _project(tmp_path, artifact=False)
    with pytest.raises(RuntimeError, match="valid configured model artifact"):
        create_app(root=root)


def test_malformed_artifact_fails_clearly(tmp_path: Path) -> None:
    root = _project(tmp_path, artifact=False)
    (root / "models" / "trained" / "api_model.joblib").write_bytes(b"not joblib")
    with pytest.raises(RuntimeError, match="Unable to initialize fraud-detection-api"):
        create_app(root=root)


def test_predict_returns_contract_and_does_not_write(tmp_path: Path) -> None:
    root = _project(tmp_path)
    before = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    with TestClient(create_app(root=root)) as client:
        response = client.post(
            "/predict", json={"features": {"amount": 0.0, "balance": 0.0}}
        )
    after = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    assert response.status_code == 200
    assert set(response.json()) == {
        "is_fraud",
        "fraud_probability",
        "threshold",
        "model_version",
    }
    body = response.json()
    assert (
        body["threshold"]
        == ModelPredictor(root / "models" / "trained" / "api_model.joblib").threshold
    )
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])
    assert body["model_version"] == "test-1"
    assert after == before


def test_predict_validation_and_payload_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    monkeypatch.setenv("FRAUD_API_MAX_BODY_BYTES", "128")
    with TestClient(create_app(root=root)) as client:
        for payload in (
            {"features": {"amount": 0.0}},
            {"features": {"amount": 0.0, "balance": 0.0, "extra": 1.0}},
            {"features": {"amount": "not-a-number", "balance": 0.0}},
        ):
            assert client.post("/predict", json=payload).status_code == 422
        assert client.post("/predict", content=b"{not-json").status_code == 422
        assert (
            client.post(
                "/predict",
                content=b'{"features":{"amount":NaN,"balance":0.0}}',
            ).status_code
            == 422
        )
        assert client.post("/predict", content=b"x" * 129).status_code == 413


def test_request_ids_and_cors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    with TestClient(create_app(root=root)) as client:
        provided = client.get("/health", headers={"X-Request-ID": "request-123"})
        generated = client.get("/health")
        preflight = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:4321",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert provided.headers["X-Request-ID"] == "request-123"
    assert generated.headers.get("X-Request-ID")
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:4321"

    monkeypatch.setenv(
        "FRAUD_API_CORS_ORIGINS", "https://demo.example, https://other.example"
    )
    with TestClient(create_app(root=root)) as client:
        custom = client.options(
            "/predict",
            headers={
                "Origin": "https://demo.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert custom.headers["access-control-allow-origin"] == "https://demo.example"


def test_unknown_route_returns_fastapi_json_404(tmp_path: Path) -> None:
    with TestClient(create_app(root=_project(tmp_path))) as client:
        response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_module_entrypoint_uses_environment_without_binding_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("fraud_detection.api.__main__")
    captured: dict[str, object] = {}
    monkeypatch.setenv("FRAUD_API_ROOT", str(_project(tmp_path)))
    monkeypatch.setenv("FRAUD_API_HOST", "0.0.0.0")
    monkeypatch.setenv("FRAUD_API_PORT", "8765")
    monkeypatch.setattr(
        module, "create_app", lambda root: captured.setdefault("app", root)
    )
    monkeypatch.setattr(
        module.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs)
    )

    module.main()

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8765
    assert captured["app"] == Path(os.environ["FRAUD_API_ROOT"])
