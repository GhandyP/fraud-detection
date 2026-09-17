# Portfolio Improvement Roadmap

This roadmap turns the current fraud-detection scaffold into a reproducible, portfolio-quality application. The browser will provide an interactive demonstration, while FastAPI will remain the only runtime boundary that loads and executes the trained scikit-learn pipeline.

## Quick path

1. Establish a clean, installable Python project.
2. Add configuration and strict data contracts.
3. Build and persist a complete, reproducible ML pipeline.
4. Expose the workflow through a CLI.
5. Expose validated inference through FastAPI.
6. Build an Astro/TypeScript interactive client.
7. Add tests, CI, documentation, and an optional deployment setup.

## Target architecture

```text
Astro + TypeScript UI
        │
        │ HTTP/JSON using the FastAPI OpenAPI contract
        ▼
FastAPI inference service
        │
        ▼
Persisted scikit-learn pipeline and metadata
```

### Responsibility boundaries

| Component | Responsibility | Explicit non-responsibility |
|-----------|----------------|-----------------------------|
| ML core | Data contracts, preprocessing, training, evaluation, and inference | UI concerns or HTTP transport |
| CLI | Dataset validation, training, evaluation, and local predictions | Browser rendering |
| FastAPI | Request validation, artifact loading, inference, health, and model metadata | Training a model per request |
| Astro/TypeScript | Forms, examples, API calls, loading/error states, and result visualization | Running Python, scikit-learn, or the model artifact |

The model artifact must never be shipped to the browser. The frontend must not duplicate preprocessing or prediction logic.

## Phases

### Phase 0 — Foundation and repository hygiene

**Goals**

- Convert the current scaffold into a standard Python package, preferably under `src/fraud_detection/`.
- Add `pyproject.toml`, a documented Python version, and one reproducible dependency workflow.
- Create clear locations for `tests/`, `docs/`, `data/`, `models/`, and `frontend/`.
- Improve `.gitignore` for caches, datasets, model artifacts, reports, environments, and frontend dependencies.
- Remove accidental tracked artifacts such as `__pycache__` files and the empty `pandas` file.
- Preserve the real dataset outside Git.

**Acceptance gate**

- A clean clone can install the project and import the package without relying on the current working directory.

### Phase 1 — Configuration and data contracts

**Goals**

- Make `config/config.yaml` the source of runtime configuration.
- Add typed configuration with safe defaults and CLI overrides.
- Resolve paths relative to the project root.
- Validate required columns, numeric types, missing values, finite values, target values, and minimum class counts.
- Define the behavior for empty datasets, one-class datasets, and invalid sample sizes.
- Add a small synthetic fixture for tests and CI.
- Document the external dataset source, license, expected schema, and privacy constraints.

**Acceptance gate**

- Valid input is accepted and invalid input fails with an actionable error message.

### Phase 2 — Reproducible ML pipeline and artifacts

Phase 2 is split so the baseline artifact contract can be delivered before the
more demanding evaluation strategy is added.

#### Phase 2A — Baseline pipeline and artifact contract — complete

**Delivered**

- Combine preprocessing and the classifier in one persisted `sklearn.Pipeline`.
- Persist the complete inference artifact with metadata:
  - model name and version;
  - ordered feature names;
  - threshold;
  - training configuration and seed;
  - dataset hash;
  - evaluated metrics;
  - Python and dependency versions;
  - creation timestamp and sample size.
- Validate raw feature schemas at prediction time and apply the persisted
  preprocessing exactly once.
- Save artifacts atomically and reject malformed or legacy bundles.

**Acceptance gate**

- Training, saving, loading, and predicting with the same artifact produces
  consistent results and applies identical preprocessing.
- Synthetic end-to-end tests cover the fixture-free workflow without tracking
  financial data or generated model files.

**Review note**

- Implementation and verification are complete. Native review was attempted,
  but the provider returned malformed collect bindings, so no native review
  evidence or approval exists for this candidate.

#### Phase 2B — Evaluation strategy and thresholding — complete

**Goals**

- Keep train, validation, and test data logically separate.
- Make the split strategy explicit and support a temporal evaluation path when
  the dataset allows it.
- Use fraud-appropriate metrics, with PR-AUC as the primary metric and recall,
  precision, F1, ROC-AUC, and a confusion matrix as supporting metrics.
- Select the classification threshold using validation data, never the final
  test set.
- Keep artifact loading restricted to a configured, trusted location.

**Delivered (P2B-01..P2B-05; committed as `3d016af` and `4ee2fef`, both pushed)**

- Deterministic three-way train/validation/test splitting with
  `validation_size`/`split_strategy` configuration and legacy two-way
  compatibility.
- Centralized fraud metrics (PR-AUC, ROC-AUC, recall, precision, F1, and a
  fixed-shape confusion matrix).
- Validation-only maximum-F1 threshold selection with highest-threshold
  tie-breaking.
- Train/validation/test orchestration with separate persisted metrics under
  artifact schema `2.0`, strict metadata validation, and schema `1.0` loading
  compatibility.
- Optional deterministic temporal evaluation with stable duplicate-timestamp
  ordering, timestamp exclusion from features, and documented leakage and
  non-stratification limitations.

**Acceptance gate**

- Threshold selection is reproducible, uses only validation data, and reports
  final metrics on an untouched test set. Met for the random workflow.
- Temporal evaluation and its limitations are covered by tests and documented.
  Met.

**Review note**

- Implementation and verification are complete. Native review evidence remains
  unavailable because the provider previously returned malformed collect
  bindings; no native review approval exists for these candidates.

### Phase 3 — Command-line interface

**Status: delivered (P3-01/P3-02)**

**Delivered**

- P3-01: `fraud` entry point, `validate-data`, and `train` with stable exit codes and JSON output.
- P3-02: trusted-artifact `evaluate`, single-record and batch `predict`, centralized metrics, documented CLI contracts, and a full synthetic fixture E2E.

**Acceptance gate**

- The complete fixture workflow runs from the CLI without manually importing Python modules. Met.

**Review note**

- Implementation and verification are complete. Native review evidence remains unavailable because the provider previously returned malformed collect bindings; no native review approval exists for this candidate.

**Goals**

Provide clear commands such as:

```text
fraud validate-data
fraud train
fraud evaluate
fraud predict
```

The CLI should:

- load the configured settings;
- validate input data before training;
- print human-readable metrics;
- support JSON output for automation;
- save model and report artifacts;
- support single-record and batch prediction;
- return useful exit codes.

**Acceptance gate**

- The complete fixture workflow runs from the CLI without manually importing Python modules.

### Phase 4 — FastAPI inference service

**Status: delivered (P4-01/P4-02)**

**Goals**

- Load the persisted pipeline once during application startup.
- Define Pydantic request and response schemas.
- Expose an initial contract:

```text
GET  /health
GET  /model-info
POST /predict
```

- Return prediction, fraud probability, threshold, and model version.
- Validate finite values, feature names, feature order, and payload size.
- Configure CORS through environment/configuration rather than hardcoding production origins.
- Add structured request/error logging and a request identifier.
- Never expose arbitrary artifact paths.
- Never train or overwrite a model from an HTTP request.

**Delivered**

- P4-01: FastAPI service skeleton with trusted artifact loading, `GET /health`,
  and `GET /model-info`, plus the `python -m fraud_detection.api` entry point.
- P4-02: `POST /predict` with Pydantic validation, request IDs, configurable
  CORS, request-body limits, and structured 4xx error handling.

**Acceptance gate**

- API tests cover successful predictions, malformed payloads, missing features, invalid values, and missing artifacts. Met.

**Review note**

- Implementation and verification are complete. Native review evidence is
  unavailable because the provider returned malformed bindings; no native
  review approval is claimed.

### Phase 5 — Astro and TypeScript interactive client

**Status: delivered (P5-01/P5-02/P5-03)**

**Delivered**

- P5-01: Astro scaffold, TypeScript API contract types, typed client, and synthetic examples.
- P5-02: Interactive feature form with preloaded examples, model metadata, prediction results, and loading, validation, server-error, and unreachable states.
- P5-03: README frontend demo instructions and final Phase 5 verification.

**Acceptance gate**

- The frontend builds independently, connects to the local API, and demonstrates a complete prediction flow. Met.

**Review note**

- Implementation and verification are complete. Native review evidence is unavailable; no native review approval is claimed.

**Dependency note**

- Three residual npm advisories remain in the development toolchain; remediation requires the breaking `astro@7` major. They are documented as non-applicable here: no base path is configured, the esbuild issue is Windows-dev-server-only on this Linux host, and images/sharp are unused.

**Goals**

- Add a separate `frontend/` Astro application.
- Use TypeScript for UI state and API calls.
- Generate or derive client types from the FastAPI OpenAPI schema.
- Provide an interactive transaction form and preloaded example transactions.
- Send requests to FastAPI using `fetch` or a generated client.
- Display loading, success, validation-error, and server-error states.
- Visualize the prediction, probability, threshold, and model metadata.
- Make the experience responsive and understandable without claiming unsupported explainability.
- Keep all ML execution on the backend; the browser must not include the model or scikit-learn.

**Acceptance gate**

- The frontend builds independently, connects to the local API, and demonstrates a complete prediction flow.

### Phase 6 — Tests, CI, and engineering quality

**Status: delivered (P6-01/P6-02)**

**Goals**

- Add unit tests for configuration, loading, validation, preprocessing, metrics, and artifact handling.
- Add integration tests for fixture → training → artifact → prediction.
- Add API tests with FastAPI's test client.
- Add frontend type checking, linting, and build checks.
- Add formatting, linting, and Python type checking.
- Add coverage reporting.
- Add GitHub Actions for backend and frontend checks.
- Ensure CI does not require the private/real dataset.

**Delivered**

- P6-01: GitHub Actions CI runs backend checks on Python 3.11 and 3.13 and frontend build, Astro check, and TypeScript checks on Node 26.
- P6-02: Coverage is integrated with pytest-cov and documented in the README, with an uncapped initial threshold and the current baseline recorded.

**Acceptance gate**

- Every push runs the relevant checks and a clean checkout passes without real financial data. Met: CI runs the relevant checks on every push from a clean checkout without the real dataset.

**Review note**

- Implementation and verification are complete. Native review evidence remains unavailable because the provider previously returned malformed collect bindings; no native review approval exists for this candidate.

### Phase 7 — Portfolio documentation

**Status: delivered (P7-01/P7-02)**

**Delivered**

- P7-01: Architecture, model card, dataset provenance, and reproduction documentation.
- P7-02: README restructured as a skimmable portfolio entry point with verified quickstart, architecture, dataset, model, CLI, API, frontend, and quality-check pointers.

**Acceptance gate**

- Met: a new reader can clone the project, understand the architecture, run the fixture demo, and identify limitations from the README and documentation.

**Review note**

- Documentation and verification are complete. Native review evidence is unavailable because the provider previously returned malformed collect bindings; no native review approval is claimed for this candidate.

### Phase 8 — Optional packaging and deployment

Only after the previous phases are stable:

- Add Docker support for FastAPI.
- Add Docker Compose for local backend/frontend integration.
- Deploy Astro as a static frontend and FastAPI separately if desired.
- Configure environment variables and health checks.
- Document the deployment process.

Kubernetes, microservices, and distributed training are intentionally out of scope for this project.

## Reviewable delivery units

Each unit should include the code, tests, and documentation that explain its behavior:

1. `chore: establish project foundation and repository hygiene`
2. `feat: add configuration and dataset validation`
3. `feat: persist reproducible training and inference artifacts`
4. `feat: add validation-based thresholding and temporal evaluation`
5. `feat: add training and prediction CLI commands`
6. `feat: expose validated model inference through FastAPI`
7. `feat: add interactive Astro prediction client`
8. `ci: enforce backend and frontend quality checks`
9. `docs: document architecture, evaluation, and model limitations`
10. `chore: add optional local deployment setup`

Do not create separate commits only for “models”, “tests”, or “frontend files”. Each commit should represent one understandable behavior and remain independently reviewable.

## Definition of done

- [ ] A clean clone installs successfully.
- [ ] The fixture can be validated, trained, evaluated, and queried.
- [ ] The persisted artifact contains preprocessing and metadata.
- [ ] The browser calls FastAPI and never executes the model.
- [ ] Invalid data and API payloads fail safely and clearly.
- [ ] Backend and frontend tests/checks pass in CI.
- [ ] No private dataset, generated model, secret, or cache is committed.
- [ ] README, architecture documentation, model card, and limitations are complete.

## Scope guardrails

- Use the real external dataset only through documented local setup.
- Use synthetic data for automated tests and CI.
- Do not claim production fraud-detection performance from a single public dataset.
- Do not add fake feature explanations; any explainability feature must be technically supported and documented.
- Keep inference centralized in FastAPI and the ML core.
- Prefer one strong baseline with rigorous evaluation over many unvalidated models.
- Deliver one phase at a time with verification before moving to the next phase.
