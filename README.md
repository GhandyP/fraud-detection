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
```

`file_name` must be a relative filename. The loader rejects invalid sample sizes
(including zero), and dataset validation rejects missing/non-finite or non-numeric
features, invalid targets, and stratified splits that cannot represent both
classes. Invalid YAML and configuration values fail with actionable errors before
model fitting.

The current programmatic entry point is:

```python
from pathlib import Path
from fraud_detection.pipeline import run_training

run_training(Path("."))
```

The CLI, FastAPI service, Astro frontend, and persisted end-to-end model artifact
remain future phases. Training with a real dataset writes generated artifacts under
`models/trained/`; no real dataset is included in this repository.

## Test

```bash
pytest
```
