# Fraud Detection Program

A Python fraud detection pipeline using scikit-learn. Phase 0 establishes an
installable package foundation; later phases will add data validation and
application interfaces.

## Structure

- `src/fraud_detection/` installable Python package
- `data/` raw and processed datasets (not included)
- `models/` trained models and artifacts (generated locally)
- `config/` configuration files
- `tests/` package smoke tests

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

## Run the training pipeline

The real dataset is required for training. Place it at
`data/raw/creditcard.csv`, then run:

```bash
python -m fraud_detection.pipeline
```

Training currently writes generated model artifacts under `models/trained/`.
This Phase 0 foundation does not yet provide a CLI, FastAPI service, or Astro
frontend.

## Test

```bash
pytest
```
