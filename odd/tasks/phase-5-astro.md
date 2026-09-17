# ODD Feature: Phase 5 Astro/TypeScript Interactive Client

## Objective

Add a separate `frontend/` Astro application that demonstrates the prediction flow: a TypeScript UI with an interactive transaction form and preloaded examples, calling the FastAPI contract over HTTP and rendering loading, validation-error, server-error, and success states with prediction, probability, threshold, and model metadata. The browser never executes the model and never receives the joblib artifact.

## Problem

The API exists but has no human-facing demo. A portfolio-grade project needs an interactive client that proves the service contract end to end and stays honest about what it shows (no unsupported explainability).

## Why

Phase 5 makes the FastAPI contract consumable and testable from a real browser environment, and its TypeScript types become evidence the OpenAPI contract is stable enough for generated clients later.

## Scope

### In scope

- `frontend/` Astro application (own `package.json`, `astro.config.*`, `tsconfig.json`, `src/`).
- TypeScript types mirroring the FastAPI contract (`/predict` request/response, `/model-info`, `/health`, error shape) hand-maintained against the API and documented; no runtime Python in the browser.
- Interactive transaction form for the artifact's feature set, preloaded example transactions, a results panel showing `is_fraud`, `fraud_probability`, `threshold`, and `model_version`, and explicit loading / validation-error / server-error / offline states.
- API base URL configurable (default `http://127.0.0.1:8000`), CORS-aligned with the service defaults.
- Build/typecheck/lint configuration (`astro check`, `tsc --noEmit`, `eslint` or the Astro default), with `node_modules` and `dist` gitignored.
- README and roadmap documentation updates.

### Out of scope

- Running Python, scikit-learn, or the model artifact in the browser.
- Shipping the model to the frontend; feature explainability claims.
- Deployment/hosting (Phase 8); authentication; service-side changes beyond CORS defaults already implemented.
- Rewriting the FastAPI contract: the frontend must consume it as-is.

## Constraints

- The frontend is a separate build; the model artifact, scikit-learn, and preprocessing never appear in `frontend/`.
- Installing npm dependencies (Astro and its toolchain) requires explicit user authorization before scaffolding.
- All tests and checks must run without the real dataset or a running API; API-dependent behavior is covered by the backend suite and by documented manual run instructions, plus optional lightweight local checks.
- `/predict` payloads must be exactly the feature map the artifact expects (finite values, exact names/order); the UI validates client-side but the API remains the authority (422 handling must be visible).
- English technical artifacts; keep each work unit reviewable and update this document and its Engram mirror after every task.
- Native review evidence is not currently available because the provider previously returned malformed collect bindings; do not claim review approval.

## TDD and verification

- Strict TDD is not active in the current session; use ordinary functional checks.
- Frontend checks per work unit: `npm run build` (Astro build), `astro check`/`tsc --noEmit` type checking, and lint as configured.
- Backend suite must stay green (`python3 -m pytest -q`).
- Temporary outputs (`node_modules`, `dist`) stay outside Git; `.astro/` cache ignored.

## Tasks

### P5-01 — Astro scaffolding, configuration, types, and API client

- **Status:** done
- **Result:** `frontend/` is a minimal Astro app (astro 5.18.2, typescript, @astrojs/check) with a client-side-only `/health` wiring page. `src/lib/types.ts` mirrors the FastAPI contract exactly (hand-maintained, documented); `src/lib/api.ts` is a typed fetch client with `PUBLIC_API_BASE_URL` (fallback `http://127.0.0.1:8000`), typed `ApiError` (status/detail/request id) and `ApiUnreachableError`; `src/lib/examples.ts` builds deterministic synthetic examples from the artifact's own feature names. `.gitignore` covers `frontend/node_modules`, `frontend/dist`, `frontend/.astro`. `npm audit fix` applied in-range updates (astro 5.13.5→5.18.2); 3 residual advisories remain whose fix requires the breaking astro@7 major: the Astro base-path advisory does not apply (no base configured), the esbuild issue is Windows-dev-server-only (Linux host), and sharp is unused (no images). Documented and flagged, not force-upgraded.
- **Observed checks:** `npm install` + `npm run build` passed; `npm run typecheck`/`npm run check` passed (0 diagnostics); backend full pytest 126 passed; `git diff --check` passed; `git status` showed only intended files; no Python/model artifacts in `frontend/`.
- **Known boundary:** Interactive UI (form, states, results panel) remains in P5-02.
- **Allowed edit surfaces:**
  - `frontend/package.json`
  - `frontend/astro.config.mjs`
  - `frontend/tsconfig.json`
  - `frontend/src/lib/types.ts`
  - `frontend/src/lib/api.ts`
  - `frontend/src/lib/examples.ts`
  - `frontend/src/index.astro`
  - `.gitignore`
- **Acceptance:** `npm run build` succeeds on an empty-ish scaffold with the typed client compiling; types match the FastAPI schemas (contract fields only); API base URL env-configurable; `node_modules`/`dist` ignored; no model/scikit code anywhere in frontend. Met.
- **Checks:** frontend build + typecheck, backend suite still green, and diff check. Passed.

### P5-02 — Interactive UI: form, examples, results, and states

- **Status:** done
- **Result:** The demo page renders model metadata from `/model-info` (version, feature count, threshold, schema version, split strategy), builds one labeled numeric input per artifact feature in artifact order, and offers preloaded example buttons generated from `buildExamples(feature_names)`. Client-side validation rejects blank, non-numeric, and non-finite values with per-field errors before any request. Submitting calls `/predict` and renders `is_fraud`, `fraud_probability`, `threshold`, and `model_version` with aria-live status banners for loading, client validation error, 422 detail, server error, and unreachable API; fields disable while requests are in flight and focus moves to the result heading on success. No UI framework, no new dependencies, no explainability claims. The parent fixed the worker's page-routing defect by moving the page to `src/pages/index.astro` (Astro emitted 0 pages from `src/`), updating import paths, and re-verifying.
- **Observed checks:** `npm run build` passed (1 page emitted; previously 0 pages before the parent's routing fix), `npm run typecheck` and `npm run check` passed with 0 diagnostics; backend full pytest 126 passed; `git diff --check` passed; no dist/node_modules staged.
- **Allowed edit surfaces:**
  - `frontend/src/index.astro` (moved to `frontend/src/pages/index.astro` by the parent)
  - `frontend/src/styles/global.css`
  - `frontend/tsconfig.json`
- **Acceptance:** the page loads model metadata from `/model-info`; the form accepts finite numeric inputs for every feature with client-side validation; preloaded examples populate the form; submitting calls `/predict`; loading, validation-error (422 detail rendered), server-error, and API-unreachable states are clearly rendered; results show `is_fraud`, `fraud_probability`, `threshold`, `model_version`; no unsupported claims. Met.
- **Checks:** frontend build + typecheck + check, backend suite green, and diff check. Passed.

### P5-03 — Documentation and full verification

- **Status:** done
- **Result:** README documents the frontend demo (API + Astro dev server startup, page behavior, synthetic-examples disclaimer, `PUBLIC_API_BASE_URL`, build/check/typecheck commands, trust boundary) and the residual npm toolchain advisories with the astro@7 remediation note; PLAN marks Phase 5 delivered with acceptance met and the standard review note. The independent verifier flagged three residual defects after the docs pass: a stale "Astro frontend remains a future phase" README paragraph, missing `npm run typecheck`/advisory note in the frontend section, and invalid CSS (`max-width: sixtyrem`); the parent fixed all three and re-verified.
- **Observed checks:** `git diff --check` passed; backend full pytest 126 passed; `npm run build` (1 page), `npm run typecheck`, and `npm run check` (0 diagnostics) passed after the fixes.
- **Allowed edit surfaces:**
  - `README.md`
  - `Doc/PLAN.md`
  - `frontend/src/styles/global.css` (parent CSS fix)
- **Acceptance:** README explains running the API + frontend locally and the trust boundary; PLAN marks Phase 5 delivered items; final build/typecheck/lint and backend suite pass. Met.
- **Checks:** final frontend build + typecheck + check, backend full suite, and diff check. Passed.

## Progress

- Phase 2B (`3d016af`, `4ee2fef`), Phase 3 CLI, and Phase 4 FastAPI were delivered; the Phase 3 + Phase 4 slices shipped together as `cac063e` (pushed).
- Node v26.8.2 / npm 11.19.1 are available on the host.
- Phase 5 npm install was authorized and executed (`frontend/` scaffolded); residual npm advisories documented as non-applicable to this deployment, remediation noted as the breaking astro@7 major.
- P5-01 (scaffold, types, API client, examples) is implemented, verified, and closed in this document and its Engram mirror.
- P5-02 (interactive UI) is implemented, verified (with the parent's page-routing fix to `src/pages/index.astro`), and closed in this document and its Engram mirror.
- P5-03 (README/PLAN docs and final verification) is closed; the parent fixed the stale README paragraph, added the typecheck/advisory documentation, and corrected the invalid CSS the verifier caught.
- Phase 5 acceptance gate is met; the slice was delivered as `f687cb2` (pushed).

## Next step

Phase 6 (tests, CI, and engineering quality) is the next roadmap unit: create `odd/tasks/phase-6-quality.md` and its Engram mirror, then implement the bounded CI/quality work units.