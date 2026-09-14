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

**Goals**

- Combine preprocessing and the classifier in one persisted `sklearn.Pipeline`.
- Keep train, validation, and test data logically separate.
- Make the split strategy explicit and support a temporal evaluation path when the dataset allows it.
- Use fraud-appropriate metrics, with PR-AUC as the primary metric and recall, precision, F1, ROC-AUC, and a confusion matrix as supporting metrics.
- Select the classification threshold using validation data, never the final test set.
- Persist the complete inference artifact with metadata:
  - model name and version;
  - ordered feature names;
  - threshold;
  - training configuration and seed;
  - dataset hash;
  - metrics;
  - Python and dependency versions;
  - creation timestamp.
- Keep artifact loading restricted to a configured, trusted location.

**Acceptance gate**

- Training, saving, loading, and predicting with the same artifact produces consistent results and applies identical preprocessing.

### Phase 3 — Command-line interface

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

Example response shape:

```json
{
  "is_fraud": false,
  "fraud_probability": 0.08,
  "threshold": 0.50,
  "model_version": "logistic-regression-v1"
}
```

**Acceptance gate**

- API tests cover successful predictions, malformed payloads, missing features, invalid values, and missing artifacts.

### Phase 5 — Astro and TypeScript interactive client

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

**Goals**

- Add unit tests for configuration, loading, validation, preprocessing, metrics, and artifact handling.
- Add integration tests for fixture → training → artifact → prediction.
- Add API tests with FastAPI's test client.
- Add frontend type checking, linting, and build checks.
- Add formatting, linting, and Python type checking.
- Add coverage reporting.
- Add GitHub Actions for backend and frontend checks.
- Ensure CI does not require the private/real dataset.

**Acceptance gate**

- Every push runs the relevant checks and a clean checkout passes without real financial data.

### Phase 7 — Portfolio documentation

**Goals**

Update the project documentation with:

- problem statement and intended use;
- architecture diagram;
- installation and local development instructions;
- dataset provenance and license;
- training, evaluation, API, and frontend commands;
- screenshots or a short demo GIF;
- reproducible results;
- model card;
- limitations, class imbalance, leakage risks, privacy, and security notes;
- example API requests and responses.

**Acceptance gate**

- A new reader can clone the project, understand the architecture, run the fixture demo, and identify the model's limitations without additional explanation.

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
4. `feat: add training and prediction CLI commands`
5. `feat: expose validated model inference through FastAPI`
6. `feat: add interactive Astro prediction client`
7. `ci: enforce backend and frontend quality checks`
8. `docs: document architecture, evaluation, and model limitations`
9. `chore: add optional local deployment setup`

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
