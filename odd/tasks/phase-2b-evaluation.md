# ODD Feature: Phase 2B Evaluation Strategy and Thresholding

## Objective

Add a reproducible Phase 2B evaluation workflow that separates train, validation, and test data; selects the classification threshold using validation only; reports fraud-appropriate metrics; and preserves the selected evaluation contract in the existing trusted model bundle.

## Problem

Phase 2A currently performs one stratified train/test split and evaluates with a fixed threshold. That baseline is suitable for artifact and inference consistency, but it cannot support honest threshold selection or distinguish validation evidence from final test evidence.

## Why

Fraud detection is sensitive to class imbalance and false-negative/false-positive tradeoffs. A validation-only threshold and explicit metrics prevent test leakage while keeping the model artifact reproducible and ready for the later CLI and FastAPI boundaries.

## Scope

### In scope

- Deterministic random train/validation/test splitting.
- Validation-only threshold selection with maximum F1 as the provisional policy.
- PR-AUC, ROC-AUC, recall, precision, F1, and fixed-shape confusion-matrix metrics.
- Separate validation and final test metrics in artifact metadata.
- Compatibility-preserving raw-feature prediction with the selected threshold.
- Optional temporal evaluation only after the random workflow is stable and its timestamp contract is explicit.
- Synthetic-only tests and documentation updates.

### Out of scope

- CLI, FastAPI, Astro, or TypeScript implementation.
- Hyperparameter search, cross-validation, grouped splitting, or production cost calibration.
- Claiming that maximum F1 is the final business policy.
- Shipping real financial data or generated model artifacts.

## Constraints

- The browser will never execute the model or receive the joblib artifact.
- The sklearn pipeline must fit preprocessing on train data only.
- Test labels must not be available to threshold selection.
- Real datasets remain external; CI uses synthetic fixtures.
- Persisted artifacts remain trusted-input-only joblib files.
- Keep each work unit reviewable and update this document after every task.
- Native review evidence is not currently available because the provider previously returned malformed collect bindings; do not claim review approval.

## TDD and verification

- Strict TDD is not active in the current session; use ordinary functional checks.
- Primary test runner: `python3 -m pytest -q`.
- Required quality checks per implementation slice: `python3 -m ruff check src tests`, `python3 -m ruff format --check src tests`, `python3 -m mypy src tests`, AST parsing, TOML parsing, and `git diff --check` as applicable.
- Temporary end-to-end smoke outputs must stay outside the repository.

## Tasks

### P2B-01 — Define split and configuration contracts

- **Status:** done
- **Result:** Added a named deterministic three-way random split while preserving the legacy two-way split contract. Added explicit validation-size and random-strategy configuration, strict proportion/class feasibility checks, and clear rejection of premature temporal settings. Original row indices remain available to prove partition disjointness.
- **Observed checks:** Focused pytest 34 passed; full pytest 59 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, and `git diff --check` passed.
- **Known boundary:** Temporal splitting remains intentionally unsupported and is reserved for P2B-05.
- **Allowed edit surfaces:**
  - `src/fraud_detection/data/preprocessor.py`
  - `src/fraud_detection/data/validation.py`
  - `src/fraud_detection/utils/config.py`
  - `config/config.yaml`
  - `tests/test_data_validation.py`
  - `tests/test_config.py`
  - `tests/test_loader.py`
- **Acceptance:** random mode produces deterministic, disjoint train/validation/test partitions; proportions and class feasibility are validated; invalid temporal requirements never silently fall back to random splitting.
- **Checks:** focused split/config tests, full pytest, lint/format/type checks, AST/TOML parsing, and diff check.

### P2B-02 — Centralize fraud metrics

- **Status:** done
- **Result:** Centralized PR-AUC, ROC-AUC, recall, precision, F1, and fixed-order confusion-matrix calculation in `models/evaluate.py`. Added validated score/threshold evaluation, deterministic zero-positive behavior, and routed `ModelTrainer.evaluate()` through the shared contract.
- **Observed checks:** Focused pytest 38 passed; full pytest 75 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, and `git diff --check` passed.
- **Known boundary:** Threshold selection remains fixed-threshold evaluation until P2B-03.
- **Allowed edit surfaces:**
  - `src/fraud_detection/models/evaluate.py`
  - `src/fraud_detection/models/train.py`
  - `tests/test_model_metrics.py`
  - `tests/test_model_artifact.py`
- **Acceptance:** metric output includes PR-AUC, ROC-AUC, recall, precision, F1, and `[[TN, FP], [FN, TP]]` counts with deterministic zero-division behavior.
- **Checks:** focused metrics tests, full pytest, lint/format/type checks, and diff check.

### P2B-03 — Select threshold from validation only

- **Status:** done
- **Result:** Added pure validation-only maximum-F1 threshold selection with inclusive score boundaries and highest-threshold tie-breaking. Added the trainer API that accepts only validation features/labels, updates the configured threshold, and persists provisional `max_f1` selection metadata.
- **Observed checks:** Focused pytest 49 passed; full pytest 86 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, and `git diff --check` passed.
- **Known boundary:** Full train/validation/test orchestration and separate final metrics remain in P2B-04.
- **Allowed edit surfaces:**
  - `src/fraud_detection/models/train.py`
  - `src/fraud_detection/models/evaluate.py`
  - `src/fraud_detection/utils/config.py`
  - `config/config.yaml`
  - `tests/test_threshold_selection.py`
- **Acceptance:** threshold selection consumes validation scores and labels only, is deterministic for ties and endpoints, maximizes validation F1, and persists the selected threshold as provisional policy metadata.
- **Checks:** threshold-focused tests proving test labels are not used, full pytest, lint/format/type checks, and diff check.

### P2B-04 — Integrate final evaluation and artifact metadata

- **Status:** done
- **Result:** `run_training()` now uses the configured three-way split: the persisted pipeline fits only on train data, the validation-only max-F1 threshold selector runs on validation scores/labels, and validation and untouched test metrics are evaluated and persisted separately under artifact schema `2.0`. `ModelTrainer.train_validate_test()` owns the sequence and `train_and_evaluate()` (schema `1.0`) remains compatible for existing callers. Save-time validation rejects partial Phase 2B arguments, inconsistent split counts vs sample size, and legacy+Phase 2B metric mixing. `ModelBundle`/`load_bundle` enforce version-specific metadata: model identity/feature/threshold consistency, training seed, strict dataset SHA-256 (hex, 64), UTC ISO-8601 timestamps, consistent runtime metadata, `random` split strategy and positive counts, validation-only `max_f1` selection, fixed metric contract, and confusion-matrix totals matching split counts. Adversarial tests prove fit isolation, inclusive persisted-threshold prediction, and parameterized malformed-metadata rejection.
- **Observed checks:** Focused pytest 67 passed; full pytest 102 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, `git diff --check`, and a temporary-directory end-to-end smoke passed.
- **Known boundary:** Temporal splitting remains intentionally unsupported; `split_strategy` is restricted to `random` until P2B-05.
- **Allowed edit surfaces:**
  - `src/fraud_detection/pipeline.py`
  - `src/fraud_detection/models/train.py`
  - `src/fraud_detection/models/artifact.py`
  - `tests/test_pipeline.py`
  - `tests/test_model_artifact.py`
- **Acceptance:** training fits only on train data; validation and untouched test metrics are stored separately; artifact schema rejects incomplete/inconsistent Phase 2B metadata; predictor uses the persisted selected threshold. Met.
- **Checks:** pipeline/artifact integration tests, full pytest, lint/format/type checks, AST/TOML parsing, smoke workflow, and diff check. Passed.

### P2B-05 — Add optional temporal evaluation and documentation

- **Status:** done
- **Result:** Implemented the approved synthetic-only temporal framework. `split_strategy: temporal` requires a non-empty `timestamp_column`; default stays `random` and rejects it. Stable `mergesort` ascending ordering with original-row-order tie policy; counts derived identically to the random path (ceil-based); oldest rows to train, then validation, newest tail to test; disjoint by construction and asserted; timestamp column excluded from features; numeric-finite/datetime64 timestamps only, missing/non-finite/string rejected; per-partition both-classes check with clear errors. Configuration parsing, `run_training()` forwarding, trainer metadata, and artifact schema `2.0` now accept `temporal` with a persisted `timestamp_column` (absent/None for random), keeping all strict metadata validation and Phase 2A compatibility. README and roadmap document temporal usage, `Time` as a possible ordering key for the external dataset (not a default, not a calendar timestamp), non-stratification, and leakage caveats.
- **Observed checks:** Focused pytest 102 passed; full pytest 109 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, `git diff --check`, and temporary-directory temporal + random E2E smokes passed; the parent fixed a markdown defect in `Doc/PLAN.md` after the worker pass.
- **Known boundary:** Temporal partitions are not stratified; chronological leakage depends on feature semantics; `Time` of the external dataset is an ordering key only.
- **Decided contract (user-approved, 2026-09-16):** synthetic-only temporal framework. `split_strategy: temporal` requires a non-empty `timestamp_column`; the default configuration remains `random`. Partitions are chronological with the same counts the random path derives (`test_count = ceil(n * test_size)`, then `validation_count = ceil((n - test_count) * validation_size / (1 - test_size))`, remainder train), assigned from the oldest rows: train, then validation, then the newest test tail. Duplicate timestamps are resolved by stable sort on the original row order (documented and tested). The timestamp column is excluded from the feature set (ordering key only). Every partition must contain both target classes with a clear error otherwise. Artifact schema `2.0` accepts `split_strategy: temporal`, persists the timestamp column in metadata, and keeps the validation-only max-F1 threshold contract. `Time` (seconds-since-first, duplicated) is documented as a possible ordering key for the real dataset, never as a default or as a real calendar timestamp. README and roadmap document limitations (no stratification in temporal mode, leakage caveats, tie policy).
- **Allowed edit surfaces:**
  - `src/fraud_detection/data/preprocessor.py`
  - `src/fraud_detection/data/validation.py`
  - `src/fraud_detection/utils/config.py`
  - `src/fraud_detection/pipeline.py`
  - `src/fraud_detection/models/train.py`
  - `src/fraud_detection/models/artifact.py`
  - `config/config.yaml`
  - `README.md`
  - `Doc/PLAN.md`
  - `tests/test_temporal_split.py`
  - `tests/test_config.py`
  - `tests/test_model_artifact.py`
  - `tests/test_pipeline.py`
- **Acceptance:** temporal partitions are chronological and disjoint; timestamp ordering, cutoffs, duplicate timestamps, class absence, and timestamp-feature policy are explicit and tested; limitations are documented.
- **Checks:** temporal-focused tests, full pytest, lint/format/type checks, AST/TOML parsing, and diff check.

## Progress

- Phase 2A baseline committed and pushed as `ab30500`.
- Phase 2B exploration completed.
- Threshold policy selected by the user: maximize validation F1 provisionally.
- P2B-01 split and configuration contracts are implemented and verified.
- P2B-02 fraud metrics are centralized and verified.
- P2B-03 validation-only threshold selection is implemented and verified.
- P2B-04 integration of the three-way workflow and strict artifact metadata is implemented, adversarially covered, and verified; the Engram mirror is synchronized.
- P2B-05 temporal framework was authorized by the user as synthetic-only (timestamp_column required, default stays random, `Time` documented as possible ordering key), implemented, adversarially verified, and closed; the Engram mirror is synchronized.

## Next step

P2B-05 was delivered as `4ee2fef` (pushed to origin/main). Open Phase 3 (CLI) as the next roadmap unit: create `odd/tasks/phase-3-cli.md` and its Engram mirror, then implement the bounded CLI work unit.
