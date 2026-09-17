# ODD Feature: Phase 4 FastAPI Inference Service

## Objective

Expose validated model inference through a FastAPI service: load the trusted artifact once at startup, validate requests with Pydantic, and serve `/health`, `/model-info`, and `/predict` with structured logging, request identifiers, and environment-configurable CORS.

## Problem

The CLI is a local operator tool. The portfolio demo and the future browser client need an HTTP boundary, and FastAPI must be the ONLY runtime that loads and executes the persisted pipeline for clients. The artifact must never reach a browser.

## Why

A FastAPI service provides the stable HTTP/JSON contract that Phase 5 (Astro frontend) will consume, reusing the ML core's validation and predictor semantics without duplicating logic and without ever training from a request.

## Scope

### In scope

- New `src/fraud_detection/api/` package (app factory, Pydantic schemas, error/request-id handling).
- Runtime dependencies `fastapi`, `uvicorn` declared in `pyproject.toml`; dev dependency `httpx` for the TestClient.
- Endpoints: `GET /health`, `GET /model-info`, `POST /predict`.
- `/predict` request: single transaction feature map with finite real values; feature names and order validated exactly against the artifact (existing `ModelPredictor` semantics); payload size limit; response `{is_fraud, fraud_probability, threshold, model_version}` plus a request identifier header.
- `/model-info`: model identity/version, ordered feature names, threshold, schema version, split strategy, timestamps.
- Request identifier (`X-Request-ID` or equivalent) and structured logging through the existing logger.
- CORS configurable through environment variables (sensible local default), never hardcoded production origins.
- Artifact path resolved only from project configuration/layout, never from request input.
- API tests with FastAPI TestClient: success, malformed payloads, missing features, invalid values, payload too large, missing artifact, unknown route.
- README and roadmap documentation updates.

### Out of scope

- Training or model overwriting from HTTP requests.
- Arbitrary artifact paths, authentication/authorization, rate limiting.
- Deployment, containers, and hosting (Phase 8).
- Astro/TypeScript frontend (Phase 5).

## Constraints

- Inference stays centralized: the browser never executes the model and never receives the joblib artifact; the API is the client boundary over the ML core.
- Installing `fastapi`, `uvicorn`, and dev `httpx` requires explicit user authorization (project constraint: no install without authorization) before implementation.
- The service loads the persisted artifact once at startup from the configured trusted location; missing/unreadable artifact is a startup or health failure with a clear status.
- No generated artifacts or data files enter the repository; API tests use synthetic fixture artifacts in temporary directories.
- English technical artifacts; keep each work unit reviewable and update this document and its Engram mirror after every task.
- Native review evidence is not currently available because the provider previously returned malformed collect bindings; do not claim review approval.
- The Phase 3 CLI slice remains uncommitted in the working tree until the user authorizes delivery; Phase 4 work stacks on top of it.

## TDD and verification

- Strict TDD is not active in the current session; use ordinary functional checks.
- Primary test runner: `python3 -m pytest -q`.
- Required quality checks per work unit: `python3 -m ruff check src tests`, `python3 -m ruff format --check src tests`, `python3 -m mypy src tests`, AST parsing, TOML parsing, and `git diff --check`.
- Temporary end-to-end smoke outputs (trained fixture artifact, logs) must stay outside the repository.

## Tasks

### P4-01 — API package skeleton, configuration, health, and model-info

- **Status:** done
- **Result:** Added `src/fraud_detection/api/` with `__init__.py`, `schemas.py`, `app.py`, and `__main__.py`. `create_app()` loads configuration and the trusted artifact once via the existing `ModelPredictor` (path derived from project layout, never from a request) and aborts with a clear actionable error when the artifact is missing or invalid. `/health` returns `{status, service}`; `/model-info` returns model identity/version, ordered feature names, threshold, schema version, split strategy, dataset hash, and creation timestamp with legacy-1.0-safe nullable fields. `python -m fraud_detection.api` runs uvicorn with `FRAUD_API_ROOT/HOST/PORT` environment defaults (127.0.0.1:8000). Pydantic v2 request model reserved for P4-02. The parent added a scoped `filterwarnings` entry for the pinned starlette TestClient/httpx deprecation; the suite is warning-clean even with `-W error::DeprecationWarning`.
- **Observed checks:** Focused pytest 5 passed; full pytest 123 passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, `git diff --check`, and temporary-directory API smoke passed.
- **Known boundary:** The `/predict` endpoint, request validation, request-ID middleware, CORS, and oversize handling remain in P4-02.
- **Allowed edit surfaces:**
  - `pyproject.toml`
  - `src/fraud_detection/api/__init__.py`
  - `src/fraud_detection/api/schemas.py`
  - `src/fraud_detection/api/app.py`
  - `src/fraud_detection/api/__main__.py`
  - `tests/test_api.py`
- **Acceptance:** the app factory builds from project configuration; `/health` returns service status; `/model-info` returns model identity/version, ordered feature names, threshold, schema version, split strategy, and timestamps from the trusted artifact; missing/invalid artifact aborts app creation with a clear actionable error; a `python -m fraud_detection.api` or documented uvicorn invocation runs the service. Met.
- **Checks:** API-focused tests, full pytest, lint/format/type checks, AST/TOML parsing, and diff check. Passed.

### P4-02 — POST /predict, validation, request ID, CORS, and error handling

- **Status:** done
- **Result:** `POST /predict` delegates to the trusted `ModelPredictor` and returns the exact roadmap contract `{is_fraud, fraud_probability, threshold, model_version}`. `PredictionRequest` enforces JSON-only feature maps, real numeric non-finite-free values, at most 256 features, and no extra keys. `RequestSizeLimitMiddleware` rejects bodies over 32 KiB (`FRAUD_API_MAX_BODY_BYTES`) with 413; `RequestIDMiddleware` accepts/generates `X-Request-ID` and logs one structured line per request (the parent reordered it outermost so 413 responses carry the header too); CORS origins come from `FRAUD_API_CORS_ORIGINS` (parent unified the env name with the `FRAUD_API_*` family; local Astro dev defaults) with `create_app(cors_origins=...)` for tests; error handlers map Pydantic/predictor errors to actionable 422 JSON and unexpected errors to 500. No training or artifact writes ever happen from HTTP.
- **Observed checks:** Focused pytest 8 passed; full pytest 126 passed; warning-strict suite passed; Ruff check, Ruff format check, mypy, AST parsing, TOML parsing, `git diff --check`, and temporary-directory smoke (contracts, 4xx/413/404, request IDs, CORS) passed.
- **Allowed edit surfaces:**
  - `src/fraud_detection/api/schemas.py`
  - `src/fraud_detection/api/app.py`
  - `tests/test_api.py`
- **Acceptance:** valid payload returns the exact response contract; malformed JSON, non-finite values, missing/extra/misordered features, wrong types, and oversize payloads return actionable 4xx errors; every request carries a request identifier header and structured logging; CORS origins come from the environment with a sensible local default; unknown routes return 404 JSON; `/predict` never trains or writes artifacts. Met.
- **Checks:** API tests covering all listed cases, full pytest, lint/format/type checks, and diff check. Passed.

### P4-03 — Documentation and full verification

- **Status:** done
- **Result:** README documents service startup (`python -m fraud_detection.api`), every endpoint with example request/response for `/predict`, request contract, error semantics, request IDs, the 32 KiB limit, all `FRAUD_API_*` environment variables, and the trust boundary; PLAN marks Phase 4 delivered with acceptance met and the standard review note (native review evidence unavailable). The worker audit caught the `FRAUD_API_CORS_ORIGINS` env-name mismatch with the code; the parent unified it in code and tests.
- **Observed checks:** `git diff --check` passed; full pytest 126 passed; `git status` showed only the expected files.
- **Allowed edit surfaces:**
  - `README.md`
  - `Doc/PLAN.md`
- **Acceptance:** README documents service startup, endpoints, example request/response, environment variables, and the artifact trust boundary; PLAN marks Phase 4 delivered items; full quality and smoke checks pass with a synthetic fixture artifact. Met.
- **Checks:** full pytest, lint/format/type checks, AST/TOML parsing, diff check, and temporary-directory service smoke. Passed.

## Progress

- Phase 2B delivered (`3d016af`, `4ee2fef`); Phase 3 CLI implemented and verified (118 full tests) but its delivery authorization is still pending in the working tree.
- Phase 4 dependencies were authorized and installed (fastapi 0.141.1, uvicorn 0.53.0, httpx 0.28.1, user-site with `--break-system-packages --user` because the system Python is PEP 668-managed; pins added to `pyproject.toml`).
- P4-01 (package skeleton, health, model-info) is implemented, warning-clean, and closed in this document and its Engram mirror.
- P4-02 (POST /predict, validation, request ID, CORS, error handling) is implemented, independently verified with no defects, and closed in this document and its Engram mirror.
- P4-03 (README/PLAN documentation) is closed; the parent unified the `FRAUD_API_CORS_ORIGINS` env name across code, tests, and docs.
- Phase 4 acceptance gate is met; delivery of the Phase 3 + Phase 4 working tree awaits user authorization.

## Next step

Await the user's delivery decision for the verified Phase 3 + Phase 4 changes (stage/commit/push require explicit authorization), then open Phase 5 (Astro frontend) as the next roadmap unit.