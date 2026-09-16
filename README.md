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
  random_state: 42
  model_name: logistic_regression
  model_version: "1.0"
  threshold: 0.5
```

`file_name` must be a relative filename. The loader rejects invalid sample sizes
(including zero), and dataset validation requires numeric finite features, non-empty
unique feature names, and a numeric target containing exactly 0 and 1. Invalid YAML
and configuration values fail with actionable errors before model fitting. The model
name, version, and threshold are persisted in the artifact; threshold optimization is
not supported.

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
Phase 2B. The CLI, FastAPI service, and Astro frontend remain future phases. No
real dataset or model artifact is included in this repository.

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
