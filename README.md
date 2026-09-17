# Fraud Detection Portfolio Demo

This repository is an educational fraud-scoring portfolio demo: it validates transaction-shaped tabular data, trains a reproducible classifier, persists an inference artifact, and exposes local CLI, API, and browser workflows. It is intended for learning and portfolio review, not for production payment decisions or other high-impact automated decisions.

## Architecture at a glance

The ML core owns validation, splitting, training, evaluation, artifact persistence, and prediction. The `fraud` CLI provides local operations; FastAPI is the only client-facing runtime that loads the trusted artifact; and the Astro frontend sends feature values to FastAPI without receiving or executing the model. See the [architecture guide](docs/architecture.md) for boundaries, data flow, the trust boundary, and the artifact contract.

Repository structure:

- `src/fraud_detection/` — installable Python package, ML core, CLI, and API
- `frontend/` — Astro and TypeScript browser demo
- `config/` — runtime configuration
- `data/` — local raw and processed data (not included)
- `models/` — locally generated trained artifacts (not committed)
- `tests/` — backend tests

## Quickstart

Requires Python 3.11–3.13. Install the package and development tools from the repository root:

```bash
python -m pip install -e ".[dev]"
```

The documented walkthrough uses a disposable synthetic fixture, so it does not require the canonical dataset. Follow the complete [reproduction guide](docs/reproduction.md), or run its fixture flow and then:

```bash
fraud validate-data
fraud train --json
fraud predict --record '{"amount": 3, "age": 22}' --json
fraud evaluate --data data/raw/transactions.csv --json
```

The same commands are available through `python -m fraud_detection.cli`. Training writes the configured artifact under `models/trained/` (the fixture example writes `models/trained/synthetic_model.joblib`). A programmatic alternative is `run_training(Path("."))` from `fraud_detection.pipeline`.

CLI commands return `0` on success, `1` for runtime or data errors, and `2` for configuration or usage errors.

## Dataset

The canonical source is Kaggle's [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), collected by Worldline and ULB's Machine Learning Group. It contains 492 frauds among 284,807 transactions (0.172%), so class imbalance matters. The canonical page states ODbL 1.0 for the database and DbCL 1.0 for its contents; verify the current terms on that canonical page before use or redistribution. The dataset is never committed; download it locally only. See the [dataset guide](docs/dataset.md).

## Model and evaluation

The persisted artifact contains a `StandardScaler` and balanced `LogisticRegression` in one sklearn pipeline. Training uses disjoint train, validation, and untouched test partitions; the maximum-F1 threshold is selected on validation only, then validation and test metrics are stored separately in artifact schema `2.0`. The metrics list is PR-AUC, ROC-AUC, recall, precision, F1, and a confusion matrix. See the [model card](docs/model-card.md) for intended use, limitations, and reporting policy.

## CLI workflow

The CLI supports `validate-data`, `train`, `evaluate`, and `predict`. `evaluate` consumes a labeled CSV, while `predict` accepts exactly one `--record` JSON object or `--input` CSV. Use `--json` for automation. The artifact includes the fitted preprocessing, ordered feature schema, threshold, metrics, dataset hash, and runtime metadata.

## FastAPI service

After training an artifact, start the local inference service:

```bash
python -m fraud_detection.api
```

It exposes `GET /health`, `GET /model-info`, and `POST /predict`. The request body is a `features` map whose names and insertion order must match the artifact. The stable response is:

```json
{
  "is_fraud": false,
  "fraud_probability": 0.08,
  "threshold": 0.50,
  "model_version": "logistic-regression-v1"
}
```

The service returns `422` for malformed or invalid prediction payloads, `413` for bodies over the configured limit, and `404` for unknown routes. Every response includes `X-Request-ID`. Configure it with `FRAUD_API_ROOT`, `FRAUD_API_HOST`, `FRAUD_API_PORT`, `FRAUD_API_MAX_BODY_BYTES`, and comma-separated `FRAUD_API_CORS_ORIGINS`; defaults include `127.0.0.1:8000` and the local Astro origins.

FastAPI is the trust boundary: it loads the artifact once, never trains or overwrites artifacts from HTTP, and keeps the artifact out of the browser. See [architecture](docs/architecture.md) for the deeper contract.

## Frontend demo

Run FastAPI and the Astro development server in separate terminals:

```bash
python -m fraud_detection.api
cd frontend && npm run dev
```

The demo runs at `http://localhost:4321`, loads model metadata, offers synthetic examples, and displays the API prediction. Set `PUBLIC_API_BASE_URL` to override the API base URL. From `frontend/`, use `npm run build`, `npm run check`, or `npm run typecheck` for local checks. The browser never receives the model artifact or runs sklearn.

Known dependency note: `npm audit` reports residual dev-toolchain advisories whose full remediation requires the breaking `astro@7` major; reassess before deployment.

## Quality checks and CI

From the repository root:

```bash
python -m pytest -q
python -m pytest --cov=fraud_detection --cov-report=term-missing
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src tests
python -m compileall -q src tests
```

The current baseline is 126 backend tests and approximately 83% branch-aware coverage. Coverage is uncapped while the baseline is established. The committed CI workflow checks Python 3.11 and 3.13 plus the Node 26 frontend build, Astro check, and TypeScript check. The workflow file is present in the working tree, but its first push is pending GitHub `workflow` scope; this is not a claim of a completed remote CI run.

## Further documentation

- [Architecture and trust boundaries](docs/architecture.md)
- [Dataset provenance and license guidance](docs/dataset.md)
- [Model card and limitations](docs/model-card.md)
- [Reproduction and quality checks](docs/reproduction.md)
