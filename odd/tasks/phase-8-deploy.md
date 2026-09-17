# ODD Feature: Phase 8 Optional Packaging and Deployment

## Objective

Add optional container packaging so a reader can run the FastAPI service and the Astro frontend with Docker/Compose, using the same environment variables and health semantics as the local workflow, and document the deployment process honestly.

## Problem

The project runs only from a prepared local environment (Python 3.11–3.13 + Node 26 + an existing trained artifact). Phase 8 is the roadmap's optional packaging unit; without it, running the demo requires manual toolchain setup.

## Why

Container packaging makes the demo reproducible for a reviewer who only has Docker. It also forces the operational contract (env vars, ports, health, artifact mounting, trust boundary) to be explicit rather than implicit in the local shell.

## Scope

### In scope

- `Dockerfile` for the FastAPI service (slim Python base, non-root user, healthcheck, uvicorn via the existing module entry point).
- `frontend/Dockerfile` as a multi-stage build (Node build stage → static nginx serve stage) with the API base URL injected at build time.
- `docker-compose.yml` wiring backend + frontend, environment variables, artifact volume (read-only), health checks, and `depends_on` ordering.
- `.dockerignore` files for both contexts (exclude data, models, caches, node_modules, dist, .git).
- `docs/deployment.md` plus a short README section and the PLAN update.

### Out of scope

- Kubernetes, microservices, distributed training, cloud-specific manifests.
- Baking datasets or trained artifacts into images.
- Authentication, TLS termination, or production hardening beyond a non-root user and read-only mounts.
- CI image builds (CI remains the Phase 6 quality workflow).

## Constraints

- Images must not contain the dataset, trained artifacts, or secrets; the artifact is mounted read-only at runtime, and training happens outside the containers via the documented CLI workflow.
- Environment variables must match the implemented service exactly: `FRAUD_API_ROOT`, `FRAUD_API_HOST`, `FRAUD_API_PORT`, `FRAUD_API_MAX_BODY_BYTES`, `FRAUD_API_CORS_ORIGINS`, and the frontend build argument `PUBLIC_API_BASE_URL`.
- The backend container must fail fast with a clear message when the artifact is missing (existing behavior); the compose file should surface that failure through the healthcheck/restart policy rather than hiding it.
- Verification in this environment is **static only**: the Docker socket is root-owned and the current user is not in the `docker` group, so image builds cannot run here. Configs must be validated by parsing and consistency review, and the docs must state plainly that `docker compose build/up` still needs to be run by the user with Docker access.
- English technical artifacts; keep each work unit reviewable and update this document and its Engram mirror after every task.
- Native review evidence is not currently available because the provider previously returned malformed collect bindings; do not claim review approval.

## TDD and verification

- Strict TDD is not active; use ordinary functional checks.
- Verification per unit: `docker compose config` (client-side parsing/merging; falls back to YAML parsing if the CLI requires the daemon), YAML/Dockerfile review against the implemented app, grep checks that documented env vars/ports/entry points exist in the code, `git diff --check`, backend suite green, and frontend build green.
- Image builds and container smoke tests are deferred to the user (documented commands); the Phase 8 report must state this explicitly.

## Tasks

### P8-01 — Backend image and Compose wiring

- **Status:** done
- **Result:** Added root `Dockerfile` (python:3.13-slim; copies `pyproject.toml`, `README.md` (required by the project readme declaration), `src/`, and `config/`; installs the package without dev extras; non-root `app` user; EXPOSE 8000; urllib-based `/health` healthcheck; CMD `python -m fraud_detection.api`; env defaults FRAUD_API_HOST=0.0.0.0 and FRAUD_API_PORT=8000). Added root `.dockerignore` excluding git metadata, caches, data, models, reports, frontend, docs, odd, and egg-info. Added `docker-compose.yml` with the `backend` service (published `${BACKEND_PORT:-8000}:8000`, env vars matching the app: FRAUD_API_ROOT=/app, HOST, PORT, MAX_BODY_BYTES default 32768, CORS defaults `http://localhost:4321,http://127.0.0.1:4321`; read-only `./models/trained:/app/models/trained:ro`; healthcheck; restart unless-stopped) and the frontend service (build arg `PUBLIC_API_BASE_URL=http://localhost:${BACKEND_PORT:-8000}`, ports 4321:80, depends_on backend service_healthy). The parent caught that `pip install .` would fail without `README.md` in the build context (setuptools readme resolution) and added the missing COPY.
- **Observed checks:** `docker compose config` rendered successfully without a daemon; compose YAML parsed; env vars greps matched `__main__.py`/`app.py`; artifact path grep matched the mount; `git diff --check` passed.
- **Allowed edit surfaces:**
  - `Dockerfile`
  - `.dockerignore`
  - `docker-compose.yml`
- **Acceptance:** backend image installs the package from `pyproject.toml` on a slim Python base, exposes port 8000, binds FRAUD_API_HOST=0.0.0.0, runs as a non-root user, defines a healthcheck against `/health`, and never bakes data/artifacts; compose mounts `./models/trained` read-only, passes the documented env vars, and orders the frontend after a healthy backend. Met (static validation; image build deferred).
- **Checks:** compose config, env var/port/entry-point greps, diff check. Passed.

### P8-02 — Frontend image, deployment docs, and closure

- **Status:** done
- **Result:** Added `frontend/Dockerfile` (node:26-alpine build stage with `npm ci` from the tracked lockfile, `PUBLIC_API_BASE_URL` build arg promoted to ENV, `npm run build`; nginx:alpine serve stage copying `dist/` to `/usr/share/nginx/html`, EXPOSE 80, wget healthcheck, foreground nginx) and `frontend/.dockerignore`. Wrote `docs/deployment.md` (prerequisites, artifact produced outside containers, build/run/teardown commands, service and environment-variable tables with the real code defaults, health-check semantics and the missing-artifact failure mode, trust boundary, and the explicit statement that images were not built in this environment with the exact commands a Docker-capable user should run). README gained a short optional-deployment section linking the doc; PLAN records Phase 8 delivered with image builds/smoke tests marked pending. The verifier flagged the vague CORS default in the deployment table; the parent replaced it with the actual default origins.
- **Observed checks:** `docker compose config` rendered; YAML parsed; `PUBLIC_API_BASE_URL` grep confirmed the build arg matches the variable the client reads; frontend build green; backend suite 126 passed; `git diff --check` passed.
- **Allowed edit surfaces:**
  - `frontend/Dockerfile`
  - `frontend/.dockerignore`
  - `docker-compose.yml`
  - `docs/deployment.md`
  - `README.md`
  - `Doc/PLAN.md`
- **Acceptance:** frontend image builds the Astro site in a Node stage and serves `dist/` statically, `PUBLIC_API_BASE_URL` injected at build time, port mapping 4321:80 documented; deployment doc covers prerequisites, commands, env vars, health checks, artifact requirements, trust boundary, and the honest unverified-builds statement; README links it; PLAN records Phase 8 status. Met (static validation; image build deferred).
- **Checks:** compose config, grep consistency, diff check, backend suite, frontend build. Passed.

## Progress

- Phases 2B–7 delivered (`3d016af`, `4ee2fef`, `cac063e`, `f687cb2`, `48da1e1`, `fb7ff0e`, `3034361`, all pushed).
- Docker 29.8.1 and Compose v5.5.1 exist on the host, but the daemon socket is root-only and the current user lacks `docker` group access, so Phase 8 verification here is static.
- P8-01 (backend image + compose wiring) and P8-02 (frontend image + docs) are implemented, statically validated, independently verified with no blocking defects, and closed in this document and their Engram mirror; the parent fixed the missing `README.md` COPY (setuptools readme) and the vague CORS default.
- Image builds and container smoke tests remain unverified because the Docker socket is root-only; the report must offer an optional real build smoke once Docker access is granted.
- Phase 8 is the last roadmap unit; delivery of the slice awaits user authorization.

## Next step

Await the user's delivery decision for the Phase 8 slice (commit/push requires explicit authorization), and offer an optional real `docker compose build/up` verification if the user grants Docker socket access. Then the roadmap is complete pending the still-unpushed CI workflow.