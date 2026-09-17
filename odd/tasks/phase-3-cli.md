# ODD Feature: Phase 3 Command-Line Interface

## Objective

Expose configuration, validation, training, evaluation, and prediction through a `fraud` console command so the complete fixture workflow runs without manually importing Python modules.

## Problem

The project currently exposes only programmatic entry points (`fraud_detection.pipeline.run_training`, `ModelPredictor`, validators). A portfolio-quality project needs a reviewable, automatable CLI that documents the workflow contract for the later FastAPI boundary.

## Why

A CLI is the first stable user-facing contract: it proves the fixture workflow end to end, enables JSON output for automation, and gives FastAPI a thin recurrence of the same validation/evaluation semantics without duplicating training logic.

## Scope

### In scope

- `fraud` console script registered in `pyproject.toml` (`[project.scripts]`).
- Subcommands: `validate-data`, `train`, `evaluate`, `predict`.
- Shared `--config` path override; `--json` output for automation.
- Single-record (`--record`, JSON) and batch (`--input`, CSV) prediction.
- Human-readable and JSON outputs; documented exit codes.
- CLI unit tests plus a synthetic fixture end-to-end workflow test.
- README and roadmap documentation updates.

### Out of scope

- FastAPI, Astro, TypeScript, deployment, and packaging beyond the console script.
- New model families, hyperparameter search, or threshold-policy changes.
- Shipping real financial data or generated model artifacts.

## Constraints

- The CLI stays thin: it calls the existing pipeline, validator, predictor, and metric helpers and never duplicates training/preprocessing logic.
- Standard-library `argparse` only; no new runtime dependencies.
- The browser will never execute the model; the CLI is a local operator tool.
- Tests use synthetic fixtures in temporary directories; generated outputs never enter the repository.
- Exit codes are documented and stable: 0 success, 1 runtime/data error, 2 configuration or usage error.
- English technical artifacts; keep each work unit reviewable and update this document and its Engram mirror after every task.
- Native review evidence is not currently available because the provider previously returned malformed collect bindings; do not claim review approval.

## TDD and verification

- Strict TDD is not active in the current session; use ordinary functional checks.
- Primary test runner: `python3 -m pytest -q`.
- Required quality checks per work unit: `python3 -m ruff check src tests`, `python3 -m ruff format --check src tests`, `python3 -m mypy src tests`, AST parsing, TOML parsing, and `git diff --check`.
- Temporary end-to-end smoke outputs must stay outside the repository.

## Tasks

### P3-01 — CLI entry point, validate-data, and train commands

- **Status:** done
- **Result:** Added `[project.scripts] fraud` and a thin `src/fraud_detection/cli.py` (stdlib argparse, strict exit-code mapping 0/1/2, `--json` output, `--sample-size` forwarding, `python -m fraud_detection.cli` support). `validate-data` resolves the configured or `--data`-overridden dataset and runs the existing validation; `train` delegates to `run_training()` and reports artifact path plus validation/test metrics from the artifact metadata, human-readable or JSON. Existing pipeline/validation logic is reused, not duplicated.
- **Observed checks:** Focused pytest 5 passed; full pytest 114 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, `git diff --check`, `--help` smoke, and temporary-directory CLI smoke (exit codes 0/1/2, JSON contract) passed.
- **Allowed edit surfaces:**
  - `pyproject.toml`
  - `src/fraud_detection/cli.py`
  - `tests/test_cli.py`
- **Acceptance:** `fraud` console script and `fraud --help` work; `validate-data` resolves the configured dataset (or `--data` override), reports success/errors, and returns the documented exit codes; `train` runs the configured fixture workflow, prints artifact path and validation/test metrics (human-readable and `--json`), and supports `--sample-size`. Met.
- **Checks:** CLI-focused tests, full pytest, lint/format/type checks, AST/TOML parsing, and diff check. Passed.

### P3-02 — evaluate and predict commands, docs, and full E2E

- **Status:** done
- **Result:** `evaluate` loads the trusted artifact via `ModelPredictor`, evaluates a labeled CSV with the centralized metric contract (`pr_auc`, `roc_auc`, `recall`, `precision`, `f1`, `confusion_matrix`) plus threshold/model version, human-readable or `--json`. `predict` supports `--record` (single JSON object) and `--input` (batch CSV, columns must match artifact features; a target column is ignored), reporting per-record prediction/is_fraud/fraud_probability/threshold/model_version with `--json`. All exit codes 0/1/2 are preserved; no training from the CLI; artifact/feature errors fail with exit 1 and usage errors with exit 2. README documents commands, examples, JSON output, and exit codes; PLAN marks Phase 3 delivered.
- **Observed checks:** Focused pytest 9 passed; full pytest 118 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, `git diff --check`, `--help`, and temporary-directory full workflow + negative-case smoke passed.
- **Allowed edit surfaces:**
  - `src/fraud_detection/cli.py`
  - `src/fraud_detection/models/predict.py` (only if a thin helper is needed)
  - `tests/test_cli.py`
  - `README.md`
  - `Doc/PLAN.md`
- **Acceptance:** `evaluate` loads a trusted artifact and evaluates a labeled CSV with the centralized metric contract; `predict` supports a single JSON record (`--record`) and a batch CSV (`--input`), printing predictions, probabilities, threshold, and model version, with `--json` output; the complete fixture workflow (validate-data → train → predict → evaluate) runs through the CLI alone. Met.
- **Checks:** CLI E2E fixture test, full pytest, lint/format/type checks, AST/TOML parsing, and diff check. Passed.

## Progress

- Phase 2B was delivered as `3d016af` and `4ee2fef` (both pushed); the roadmap marks Phase 2B complete.
- Phase 3 CLI is the next roadmap unit; exploration of the existing programmatic entry points is done.
- P3-01 (entry point, validate-data, train) is implemented, independently verified, and closed in this document and its Engram mirror.
- P3-02 (evaluate, predict, docs, full fixture E2E) is implemented, independently verified, and closed in this document and its Engram mirror.
- Phase 3 acceptance gate (complete fixture workflow runs from the CLI without importing Python modules) is met; delivery of the slice awaits user authorization.

## Next step

Await the user's delivery decision for the verified Phase 3 slice (stage/commit/push require explicit authorization), then open Phase 4 (FastAPI inference service) as the next roadmap unit.