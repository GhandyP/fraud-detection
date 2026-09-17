# Fraud Detection Program

A Python fraud detection pipeline using scikit-learn. Phase 1 activates typed YAML
configuration and deterministic dataset validation.

## Structure

- `src/fraud_detection/` installable Python package
- `data/` raw and processed datasets (not included)
- `models/` trained models and artifacts (generated locally)
- `config/` configuration files
- `tests/` package and behavior tests

## Installation

Requires Python 3.10–3.13.

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

`requirements.txt` remains as a compatibility install path and delegates to
`pyproject.toml`.

## Configuration and training

Training loads `config/config.yaml` relative to the project root. The expected
fields are:

```yaml
dataset:
  file_name: creditcard.csv
  target_column: Class
train:
  test_size: 0.2
  validation_size: 0.2
  split_strategy: random
  random_state: 42
  model_name: logistic_regression
  model_version: "1.0"
  threshold: 0.5
```

`file_name` must be a relative filename. The loader rejects invalid sample sizes
(including zero), and dataset validation requires numeric finite features, non-empty
unique feature names, and a numeric target containing exactly 0 and 1. Invalid YAML
and configuration values fail with actionable errors before model fitting. The model
name, version, and threshold are persisted in the artifact; threshold selection uses validation data only and is persisted with the artifact.

For synthetic or externally supplied datasets, an optional chronological split can be
configured with `split_strategy: temporal` and a non-empty `timestamp_column`:

```yaml
train:
  split_strategy: temporal
  timestamp_column: event_time
```

Temporal mode sorts ascending with a stable original-row-order tie policy, assigns
oldest rows to train, then validation, and the newest rows to test, and excludes the
timestamp column from model features. It does not stratify partitions, so each
partition must contain both classes. Chronological splits can still leak information
when features encode future knowledge; choose the ordering key and leakage controls
carefully. For the documented external credit-card dataset, `Time` is a possible
ordering key: it is seconds since the first transaction with duplicates, not a real
calendar timestamp, and it is not the default. Random mode remains the default and
rejects `timestamp_column`.

## CLI workflow

The `fraud` command exposes the local fixture workflow without importing Python
modules. Commands return `0` on success, `1` for runtime or data errors, and `2`
for configuration or usage errors.

```bash
fraud validate-data
fraud train --json
fraud predict --record '{"amount": 12.5, "age": 42}' --json
fraud predict --input predictions.csv --json
fraud evaluate --data labeled.csv --json
```

`evaluate` loads the configured model artifact (or `--model PATH`) and requires a
labeled CSV containing both target classes. `predict` requires exactly one of
`--record` or `--input`; batch columns must match the artifact feature schema (a
configured target column is ignored when present). JSON prediction output contains
`predictions`, `threshold`, and `model_version`; each prediction includes its
classification and fraud probability. Evaluation JSON reports `pr_auc`, `roc_auc`,
`recall`, `precision`, `f1`, and a 2x2 `confusion_matrix`, plus model metadata.

The current programmatic entry point is:

```python
from pathlib import Path
from fraud_detection.pipeline import run_training

run_training(Path("."))
```

Training writes a persisted Phase 2A end-to-end scikit-learn artifact under
`models/trained/`. The artifact contains the fitted scaler and balanced logistic
regression pipeline, ordered feature schema, threshold, evaluated metrics, dataset
hash, versions, timestamp, and sample-size metadata. Prediction loads this trusted
joblib artifact and accepts raw numeric finite features without manual scaling;
feature order must match exactly. Threshold optimization and advanced evaluation (validation splits,
cross-validation, temporal splits, and related metrics) are explicitly owned by
Phase 2B. The CLI and FastAPI service provide local operator and client-facing inference
workflows, and the Astro frontend demonstrates the prediction flow in the browser. No
real dataset or model artifact is included in this repository.

## FastAPI service

FastAPI is the client-facing inference boundary. Start it from the project root
after training a model artifact:

```bash
python -m fraud_detection.api
```

The entry point loads the configured artifact once at startup and runs Uvicorn.
Configure it with environment variables:

- `FRAUD_API_ROOT`: project root (defaults to the current directory).
- `FRAUD_API_HOST`: bind host (defaults to `127.0.0.1`).
- `FRAUD_API_PORT`: bind port (defaults to `8000`).
- `FRAUD_API_MAX_BODY_BYTES`: maximum request body size (defaults to 32 KiB).
- `FRAUD_API_CORS_ORIGINS`: comma-separated allowed browser origins. For local
  Astro development, use `http://localhost:4321,http://127.0.0.1:4321`.

The service exposes:

- `GET /health` — liveness and service status.
- `GET /model-info` — trusted model identity, version, ordered feature names,
  threshold, artifact schema, split, dataset, and creation metadata.
- `POST /predict` — validates one transaction and returns its prediction.

A prediction request contains a `features` map whose names and insertion order
must exactly match the artifact schema. Values must be finite real numbers. For
example, an artifact with features `amount` and `balance` can receive:

```json
{
  "features": {
    "amount": 12.5,
    "balance": 240.0
  }
}
```

The response has the stable contract:

```json
{
  "is_fraud": false,
  "fraud_probability": 0.08,
  "threshold": 0.50,
  "model_version": "logistic-regression-v1"
}
```

Malformed JSON, invalid values, or feature names/order that do not match the
artifact return `422`. Bodies over the configured 32 KiB limit return `413`,
and unknown routes return `404`. Every response includes an `X-Request-ID`
header, echoed from the request when supplied or generated by the service.

FastAPI is the only client-facing runtime that loads the model artifact. The
browser never receives the artifact or executes the model. The service never
trains models or overwrites model artifacts.

## Frontend demo

Run the FastAPI service and the Astro development server in separate terminals:

```bash
python -m fraud_detection.api
cd frontend && npm run dev
```

The frontend is available at `http://localhost:4321`, matching the service's
local CORS defaults. The page loads model metadata from `/model-info`, builds a
dynamic numeric feature form, provides preloaded synthetic examples, and shows
prediction results containing `is_fraud`, `fraud_probability`, `threshold`, and
`model_version`. Examples are synthetic; the demo reports model output without
explanations.

Set `PUBLIC_API_BASE_URL` to override the default API base URL. Build and
validate the frontend with:

```bash
cd frontend
npm run build
npm run check
npm run typecheck
```

The browser never receives the model artifact: all inference runs in FastAPI.

Known dependency note: `npm audit` reports residual advisories in the dev
toolchain (astro/esbuild/sharp) whose fix requires the breaking `astro@7`
major. They do not affect this static demo (no configured base path, Linux
development host, and no image processing); re-evaluate before deployment.

## Verified workflow

```bash
pytest
ruff check .
ruff format --check .
mypy src
python -m compileall -q src tests
git diff --check
```

The temporary integration tests exercise training, atomic artifact persistence, loading,
and raw-feature prediction without adding data or model files to the repository.
