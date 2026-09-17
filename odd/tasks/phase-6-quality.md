# ODD Feature: Phase 6 Tests, CI, and Engineering Quality

## Objective

Enforce backend and frontend quality on every push with GitHub Actions, keep the checks runnable from a clean checkout without the real dataset, and add coverage reporting for the Python suite.

## Problem

All checks currently run manually. A portfolio-grade repository needs CI gates on push/PR so regressions in the backend (pytest, Ruff, mypy, compileall) and the frontend (build, astro check, tsc) are caught automatically, with honest coverage visibility.

## Why

Phase 6 closes the engineering-quality loop: the suite (126 backend tests, frontend build/typecheck) is strong but only exists locally; CI makes the acceptance gate enforceable for every push.

## Scope

### In scope

- `.github/workflows/ci.yml` with two jobs:
  - `backend`: checkout, Python setup (3.10 and 3.13 matrix), `pip install -e ".[dev]"`, `ruff check src tests`, `ruff format --check src tests`, `mypy src tests`, `python -m compileall -q src tests`, `python -m pytest` (with coverage if authorized), all without the real dataset.
  - `frontend`: Node toolchain matching the project, `npm ci`, `npm run build`, `npm run check`, `npm run typecheck`.
- Optional (pending user authorization): `pytest-cov` dev dependency and coverage reporting (terminal summary in CI; HTML artifact optional).
- Documentation of the CI workflow in README/PLAN.

### Out of scope

- Deployment, containers, release automation (Phase 8).
- Frontend E2E/browser automation beyond build and type checks.
- Secrets, publishing, or matrix expansion beyond documented Python versions.

## Constraints

- CI must pass without the real dataset or any tracked generated artifacts (existing synthetic fixtures guarantee this).
- Adding `pytest-cov` to dev dependencies requires explicit user authorization (no install without authorization); without it, CI runs the existing checks only.
- Node version in CI should match the documented local toolchain (currently Node 26/astro 5).
- English technical artifacts; keep each work unit reviewable and update this document and its Engram mirror after every task.
- Native review evidence is not currently available because the provider previously returned malformed collect bindings; do not claim review approval.

## TDD and verification

- Strict TDD is not active; use ordinary functional checks.
- Work units verify with the same commands CI runs, plus `git diff --check` and a YAML syntax validation of the workflow file.
- Backend suite must stay green; frontend build/check/typecheck must stay green.

## Tasks

### P6-01 — CI workflow for backend and frontend

- **Status:** done
- **Result:** Added `.github/workflows/ci.yml` (triggers: push to main + pull requests; `permissions: contents: read`). Backend job: Python 3.11/3.13 matrix, `pip install -e ".[dev]"`, then the exact local checks — Ruff check, Ruff format check, mypy, compileall, and `pytest --cov=fraud_detection --cov-report=term-missing`. Frontend job: Node 26 with npm cache keyed on `frontend/package-lock.json`, `npm ci`, build, `astro check`, and `tsc --noEmit`. No dataset, secrets, or artifact downloads required.
- **Observed checks:** YAML parse passed; the workflow was independently audited against a clean-checkout simulation; backend suite, frontend build, and `git diff --check` passed.
- **Allowed edit surfaces:**
  - `.github/workflows/ci.yml`
  - `pyproject.toml`
  - `frontend/package.json` (not needed; unchanged)
- **Acceptance:** the workflow runs backend checks (Ruff, mypy, compileall, pytest, coverage) on the supported Python matrix, and frontend build/check/typecheck on Node 26; jobs fail on any error; no dataset or secrets required; workflow YAML parses cleanly. Met.
- **Checks:** YAML parse, backend suite, frontend build, and diff check. Passed.

### P6-02 — Coverage reporting and documentation

- **Status:** done
- **Result:** `pytest-cov>=6,<8` added to dev dependencies (authorized install; pytest-cov 7.1.0), `[tool.coverage.run]` with `source = ["fraud_detection"]` and `branch = true`, `[tool.coverage.report]` with `show_missing = true` and an explicitly uncapped initial threshold (branch-aware baseline 83% recorded in the README). README documents the CI workflow and the local coverage command; PLAN marks Phase 6 delivered with acceptance met and the standard review note.
- **Observed checks:** full pytest with coverage (126 passed, 83% branch-aware), Ruff, mypy, YAML parse, `git diff --check`, and frontend build passed.
- **Known boundary:** The verifier caught a credibility defect — `numpy==2.4.1` has no Python 3.10 wheels, contradicting the previous `requires-python >=3.10` and the initial CI matrix. The parent confirmed with `pip download --python-version` probes (3.10 unsupported; 3.11/3.12/3.13 supported) and raised the documented support to Python 3.11–3.13: `requires-python`, mypy target version, CI matrix, README install line and CI description, and PLAN. Ruff then flagged `UP017` for the 3.11 target; the autofix applied the `datetime.UTC` alias. All checks re-verified green.
- **Allowed edit surfaces:**
  - `pyproject.toml`
  - `.github/workflows/ci.yml`
  - `README.md`
  - `Doc/PLAN.md`
- **Acceptance:** coverage measured via `pytest-cov` with a documented uncapped initial threshold; CI prints coverage; README documents the CI workflow and coverage command; PLAN marks Phase 6 delivered items. Met.
- **Checks:** full pytest with coverage, lint/type checks, YAML validation, and diff check. Passed.

## Progress

- Phase 2B, 3, 4, and 5 delivered (`3d016af`, `4ee2fef`, `cac063e`, `f687cb2`, all pushed).
- Local quality checks are established and green (126 backend tests, frontend build/check/typecheck).
- Coverage tooling was authorized and installed (pytest-cov 7.1.0); the project's supported Python range was corrected to 3.11–3.13 (numpy 2.4.1 has no 3.10 wheels).
- P6-01 (CI workflow) and P6-02 (coverage + docs) are implemented, independently verified, and closed in this document and their Engram mirror.
- Phase 6 acceptance gate is met; delivery of the slice awaits user authorization.

## Next step

Await the user's delivery decision for the verified Phase 6 slice (stage/commit/push require explicit authorization), then open Phase 7 (portfolio documentation) as the next roadmap unit.