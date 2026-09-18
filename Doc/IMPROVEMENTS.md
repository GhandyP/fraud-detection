# Improvement Backlog

Deferred, optional, and known-limitation items collected while delivering the roadmap in `PLAN.md`. Nothing here blocks the current deliverables; each entry states what to do, why it was deferred, and where the evidence lives.

## 1. Verification gaps

### 1.1 Docker images never built

- **What:** run `docker compose build` and `docker compose up`, then verify `curl http://localhost:8000/health` and load `http://localhost:4321`.
- **Why deferred:** the Docker socket is root-owned and the working user is not in the `docker` group, so builds could not run in the delivery environment.
- **Next step:** grant socket access (`sudo usermod -aG docker $USER` plus re-login) or run the documented commands manually.
- **Evidence:** `docs/deployment.md`, `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`.

### 1.2 No native review evidence

- **What:** obtain a completed native review for a candidate (lineage + admitted verdict), starting with the next source change.
- **Why deferred:** the provider returned malformed collect bindings for every attempted candidate; no approval exists and none is claimed.
- **Next step:** retry the native review lifecycle on a fresh candidate when the provider issue is resolved.
- **Evidence:** review notes in `Doc/PLAN.md` per phase.

### 1.3 No metrics published from the canonical dataset

- **What:** run the full workflow against the canonical Kaggle dataset locally and document reproducible results (dataset SHA-256, validation and test metrics, selected threshold).
- **Why deferred:** the dataset is external and must not be redistributed; only synthetic fixtures run in tests and CI.
- **Next step:** after downloading the data, record the artifact metadata (it already embeds the hash and metrics) in a `docs/results.md`-style report.
- **Evidence:** `docs/dataset.md`, `docs/reproduction.md`, artifact schema `2.0` metadata.

## 2. Dependency and tooling debt

### 2.1 Residual npm advisories

- **What:** upgrade Astro to the `astro@7` major (or later) and re-verify the frontend build, `astro check`, and `tsc`.
- **Why deferred:** the fix is a breaking major upgrade; the current advisories do not apply to this static demo (no configured base path, Linux development host, no image processing).
- **Next step:** schedule the upgrade with a full frontend verification pass and update `docs/reproduction.md` if tooling versions change.
- **Evidence:** README "Frontend demo" note, `frontend/package.json`, `odd/tasks/phase-5-astro.md`.

### 2.2 Coverage threshold uncapped

- **What:** set `fail_under` in `[tool.coverage.report]` once the suite stabilizes around a deliberate floor.
- **Why deferred:** the initial Phase 6 decision recorded an explicitly uncapped threshold with an 83% branch-aware baseline.
- **Next step:** choose a floor (for example 80) and add it with the next backend change.
- **Evidence:** `pyproject.toml`, README "Quality checks and CI".

### 2.3 `requirements.txt` duplication

- **What:** decide whether to keep the compatibility shim or delete it in favor of `pyproject.toml` alone.
- **Why deferred:** it exists as a convenience path for tooling that expects a requirements file.
- **Next step:** remove it if nothing consumes it; otherwise document its maintenance rule.
- **Evidence:** `requirements.txt`, README installation section.

### 2.4 Git author identity

- **What:** configure a proper `user.name` / `user.email` for the repository or machine.
- **Why deferred:** commits currently use the auto-detected identity (`vit@vit.vit`), which is not portfolio-appropriate.
- **Next step:** `git config --global user.name "..."` and `user.email "..."`.
- **Evidence:** the committer warning printed by every local commit.

## 3. Model and evaluation

### 3.1 Cost-sensitive threshold policy

- **What:** replace or complement maximum-F1 selection with an explicit business-cost objective (for example expected monetary loss with a false-negative-to-false-positive cost ratio).
- **Why deferred:** maximum F1 is the provisional policy recorded in Phase 2B; cost calibration needs domain inputs the project does not have.
- **Next step:** add a cost matrix to configuration and extend `select_threshold` with a named policy, keeping validation-only selection.
- **Evidence:** `src/fraud_detection/models/evaluate.py`, `odd/tasks/phase-2b-evaluation.md`.

### 3.2 Model comparison and cross-validation

- **What:** compare the baseline against alternatives (gradient boosting, calibrated linear models) with cross-validation and PR-AUC-first reporting.
- **Why deferred:** the roadmap deliberately preferred one rigorously evaluated baseline over many unvalidated models.
- **Next step:** new ODD feature document; keep the artifact contract and evaluation protocol unchanged.
- **Evidence:** `docs/model-card.md`, `Doc/PLAN.md` scope guardrails.

### 3.3 Temporal evaluation with real timestamps

- **What:** exercise the temporal split strategy on data with a meaningful calendar timestamp, including leakage analysis on features that encode future knowledge.
- **Why deferred:** the canonical dataset only has `Time` (seconds since the first transaction, with duplicates), so temporal mode is synthetic-tested and documented rather than demonstrated.
- **Next step:** add a dataset with real timestamps or a documented feature policy for `Time`.
- **Evidence:** `docs/dataset.md`, `odd/tasks/phase-2b-evaluation.md` (P2B-05).

### 3.4 Drift and monitoring

- **What:** define input-drift checks (feature distributions, prediction score distributions) and a retraining trigger policy.
- **Why deferred:** monitoring requires a deployed environment, which is out of the roadmap scope.
- **Next step:** specify metrics and thresholds when a real deployment exists.
- **Evidence:** `docs/model-card.md` monitoring notes.

## 4. API and frontend hardening

### 4.1 API security and limits

- **What:** add authentication, per-client rate limiting, and TLS termination guidance before any non-local exposure.
- **Why deferred:** the service is a local demo boundary; the current scope is validation, request IDs, CORS, and a 32 KiB body cap.
- **Next step:** decide the deployment model first (see `docs/deployment.md`), then add auth and limits.
- **Evidence:** `src/fraud_detection/api/app.py`, `docs/architecture.md`.

### 4.2 Generated TypeScript types

- **What:** generate the frontend contract types from the FastAPI OpenAPI schema (for example `openapi-typescript`) and add a drift check.
- **Why deferred:** types are hand-maintained and documented as such; the contract is small and stable.
- **Next step:** add the generator and a CI comparison step; keep the hand-written client wrapper.
- **Evidence:** `frontend/src/lib/types.ts`, `frontend/src/lib/api.ts`.

### 4.3 Frontend demonstration assets

- **What:** add screenshots or a short demo GIF to the README and `docs/`.
- **Why deferred:** no browser session was available in the delivery environment, and fabricating visuals is unacceptable.
- **Next step:** run the stack locally and capture real images, including at least one validation-error state.
- **Evidence:** `docs/reproduction.md` honest no-screenshots note.

### 4.4 Frontend lint/format tooling

- **What:** add ESLint/Prettier (or equivalent) alongside `astro check` and `tsc`.
- **Why deferred:** Phase 5 kept the dependency footprint minimal.
- **Next step:** add the tooling, wire it into `npm run check` and CI.
- **Evidence:** `frontend/package.json`, `.github/workflows/ci.yml`.

## 5. Repository and process

### 5.1 Missing repository LICENSE

- **What:** add a license for the repository's own code (the dataset license is separate and documented in `docs/dataset.md`).
- **Why deferred:** never specified in the roadmap; noted here as a portfolio gap.
- **Next step:** choose a license (for example MIT) and add `LICENSE` plus a README line.
- **Evidence:** repository root, `docs/dataset.md`.

### 5.2 CI extensions

- **What:** publish the coverage summary as a build artifact or a badge, enable dependency scanning (Dependabot), and optionally build container images in CI.
- **Why deferred:** Phase 6 scoped CI to the quality gates; image builds depend on section 1.1.
- **Next step:** extend `.github/workflows/ci.yml` once the Docker builds are verified locally.
- **Evidence:** `.github/workflows/ci.yml`, CI run 35280699741.

### 5.3 Contract tests between API and frontend types

- **What:** add a test that compares the hand-maintained TypeScript contract types against the FastAPI schemas.
- **Why deferred:** the schemas are stable and the drift risk is currently low.
- **Next step:** pair with section 4.2 (generated types make this obsolete).
- **Evidence:** `src/fraud_detection/api/schemas.py`, `frontend/src/lib/types.ts`.

### 5.4 Roadmap-complete next steps

- **What:** if the roadmap is extended, open a new ODD feature document before writing code (for example explainability, batch scoring, or a second model).
- **Why deferred:** the roadmap in `Doc/PLAN.md` is fully delivered.
- **Next step:** create `odd/tasks/<feature>.md` plus its Engram mirror and follow the same phase discipline.
- **Evidence:** `Doc/PLAN.md`, `odd/tasks/`.