# ODD Feature: Phase 7 Portfolio Documentation

## Objective

Turn the repository into a self-explanatory portfolio project: a new reader should clone it, understand the architecture and trust boundaries, run the fixture workflow, and recognize the model's limitations without extra explanation.

## Problem

The README grew phase by phase and now mixes quickstart, CLI, API, and frontend details while the deeper material (architecture rationale, model card, dataset provenance, limitations, security) lives nowhere. The roadmap's Phase 7 acceptance gate requires that material to be explicit and accurate.

## Why

Reviewers of a portfolio project check three things fast: what the project claims, whether the claims are honest, and whether they can reproduce the demo. Documentation is the deliverable that makes the other phases legible.

## Scope

### In scope

- `docs/architecture.md`: component boundaries (ML core, CLI, FastAPI, Astro), data flow, trust boundary, artifact schema summary, and example API request/response.
- `docs/model-card.md`: intended use, out-of-scope uses, model family and training configuration, evaluation protocol (three-way split, validation-only max-F1 threshold, untouched test metrics), metrics reporting policy, known biases, and monitoring/limitations notes.
- `docs/dataset.md`: canonical provenance (Kaggle `mlg-ulb/creditcardfraud`, ULB Machine Learning Group + Worldline), canonical paper citation, schema, class imbalance, license statement with verified evidence and a "confirm on the canonical page" caveat, and the no-redistribution rule.
- `docs/reproduction.md`: install, fixture workflow (synthetic data only), CLI/API/frontend commands, CI workflow summary, coverage baseline, and the honest note that visual screenshots/GIFs are not committed.
- README restructuring: problem statement and intended use up front, quickstart, short architecture summary linking to `docs/`, dataset pointer, limitations summary, and removal of duplicated deep detail.

### Out of scope

- Code, test, or contract changes.
- Fabricated screenshots, GIFs, or benchmark claims.
- Deployment docs (Phase 8).

## Constraints

- Every claim must be traceable to repository code/config or to a cited external source; no invented metrics, licenses, or explainability.
- Dataset license wording must reflect verified evidence: the canonical Kaggle dataset page states ODbL 1.0 with contents under DbCL 1.0 (corroborated by mirrored quotes and secondary sources); the docs must instruct readers to confirm on the canonical page and must not redistribute data.
- Metrics must be described as what the code actually does: validation-only max-F1 threshold selection, validation and untouched test metrics persisted separately in artifact schema `2.0`, PR-AUC first-class alongside ROC-AUC/recall/precision/F1/confusion matrix.
- Screenshots/GIFs: not produced in this environment; document how to see the demo and what it shows instead of fabricating visuals.
- English technical artifacts; keep each work unit reviewable and update this document and its Engram mirror after every task.
- Native review evidence is not currently available because the provider previously returned malformed collect bindings; do not claim review approval.

## TDD and verification

- Strict TDD is not active; use ordinary functional checks.
- Docs units verify with: link/path existence checks, `git diff --check`, the backend suite (must stay green), and frontend build when README commands change.
- Any documented command must exist in the repository (scripts, module entry points) or be explicitly marked as not available.

## Tasks

### P7-01 — Architecture, model card, dataset, and reproduction docs

- **Status:** done
- **Result:** Wrote `docs/architecture.md` (problem framing, component/boundary table, Mermaid data-flow diagram, trust boundary, artifact schema 2.0 summary with legacy 1.0 note, exact `/predict` contract example), `docs/model-card.md` (intended and out-of-scope uses, model family, training configuration, validation-only max-F1 protocol, reporting policy, limitations, monitoring notes), `docs/dataset.md` (canonical Kaggle `mlg-ulb/creditcardfraud` provenance, schema table, ODbL/DbCL statement with verify-on-canonical-page instruction, non-canonical Zenodo copy flagged, Dal Pozzolo et al. CIDM 2015 citation, no-redistribution rules), and `docs/reproduction.md` (prerequisites, install, synthetic walkthrough, local services, quality checks, CI summary, coverage baseline, explicit no-screenshots statement).
- **Observed checks:** four documents present; link/path audit passed (runtime-only `data/raw/` and `models/trained/` correctly described); `git diff --check` passed; backend suite 126 passed.
- **Allowed edit surfaces:**
  - `docs/architecture.md`
  - `docs/model-card.md`
  - `docs/dataset.md`
  - `docs/reproduction.md`
- **Acceptance:** the four documents exist, are consistent with the implemented code, cite the canonical dataset source and paper, state the trust boundary and limitations honestly, and contain no fabricated visuals or metrics. Met.
- **Checks:** internal link/path verification, `git diff --check`, backend suite green. Passed.

### P7-02 — README restructuring and roadmap closure

- **Status:** done
- **Result:** README restructured into a portfolio entry point (problem statement and intended use, architecture at a glance, quickstart, dataset summary, model and evaluation summary, CLI/API/frontend sections retaining exact contracts and env vars, quality/CI section with the pending-workflow disclosure, and links to all four docs); `Doc/PLAN.md` marks Phase 7 delivered with the acceptance gate met and the standard review note. The independent verifier caught a malformed install command (`pip install -e "[dev]"`) in `docs/reproduction.md`; the parent fixed it to `-e ".[dev]"` and re-verified.
- **Observed checks:** documented CLI/API/frontend/pyproject entries grep-verified against the repository; backend suite 126 passed; frontend build green; `git diff --check` passed.
- **Allowed edit surfaces:**
  - `README.md`
  - `Doc/PLAN.md`
  - `docs/reproduction.md` (parent install-command fix)
- **Acceptance:** README opens with problem statement and intended use, provides a quickstart, links to the `docs/` set, keeps commands accurate, and avoids duplicating deep detail; PLAN marks Phase 7 delivered. Met.
- **Checks:** documented commands verified against the repository, `git diff --check`, backend suite green, frontend build green. Passed.

## Progress

- Phases 2B–6 delivered (`3d016af`, `4ee2fef`, `cac063e`, `f687cb2`, `48da1e1`, `fb7ff0e`).
- Phase 6 follow-up pending: `.github/workflows/ci.yml` is implemented and verified but unpushed (OAuth token lacks the `workflow` scope), and the parent found that `frontend/package-lock.json` is currently gitignored, which would break `npm ci` in CI; both must be fixed together in the follow-up CI commit.
- Dataset provenance and license evidence was gathered with cited sources before writing (canonical Kaggle dataset page terms, corroborating mirrors, Zenodo re-upload noted as a non-canonical copy with different terms).
- P7-01 (docs set) and P7-02 (README/PLAN) are implemented, independently verified, and closed in this document and their Engram mirror; the parent fixed the malformed install command the verifier caught.
- Phase 7 acceptance gate is met; delivery awaits user authorization.

## Next step

Await the user's delivery decision for the verified Phase 7 documentation slice (commit/push requires explicit authorization). Separately, the pending Phase 6 follow-up (tracking `frontend/package-lock.json` and pushing `.github/workflows/ci.yml`) still waits on the GitHub `workflow` scope; the lockfile fix alone does not require that scope and can ship with the docs commit.