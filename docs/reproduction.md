# Reproduction

## Prerequisites and installation

- Python 3.11–3.13
- Node 26 for the Astro frontend

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

The default configuration expects `data/raw/creditcard.csv`. The canonical dataset is not included; see [Dataset](dataset.md). The walkthrough below uses a disposable synthetic fixture instead.

## Complete synthetic-fixture walkthrough

Create a temporary project with the same two numeric features used by the CLI tests. The fixture is deliberately synthetic and must not be interpreted as transaction data:

```bash
DEMO_DIR="$(mktemp -d)"
mkdir -p "$DEMO_DIR/config" "$DEMO_DIR/data/raw"
DEMO_DIR="$DEMO_DIR" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["DEMO_DIR"])
(root / "config/config.yaml").write_text("""dataset:
  file_name: transactions.csv
  target_column: label
train:
  test_size: 0.2
  validation_size: 0.2
  split_strategy: random
  random_state: 42
  model_name: synthetic_model
  model_version: 'synthetic-1'
  threshold: 0.5
""", encoding="utf-8")
pd.DataFrame({
    "amount": list(range(1, 21)),
    "age": list(range(20, 40)),
    "label": [0] * 10 + [1] * 10,
}).to_csv(root / "data/raw/transactions.csv", index=False)
print(root)
PY
cd "$DEMO_DIR"
fraud validate-data
fraud train --json
fraud predict --record '{"amount": 3, "age": 22}' --json
fraud evaluate --data data/raw/transactions.csv --json
```

Training creates the local `models/trained/synthetic_model.joblib` artifact in the temporary project. The JSON output contains validation and test metrics; the values depend on the fixture and are not portfolio benchmark claims.

## See the local API and frontend demo

From the synthetic project directory, after training:

```bash
python -m fraud_detection.api
```

The service listens on `http://127.0.0.1:8000` by default. It exposes `GET /health`, `GET /model-info`, and `POST /predict`. In a second terminal, from the repository's `frontend/` directory:

```bash
cd frontend
PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm install
npm run dev
```

Open `http://localhost:4321`. The page loads metadata, creates a numeric form from the artifact feature names, offers synthetic examples, and displays the API's fraud status, probability, threshold, and model version. It displays model output only; it does not provide explanations. No model artifact is sent to the browser.

For a clean dependency install when the lockfile is available, use `npm ci` instead of `npm install`.

## Quality checks

Backend checks, from the repository root:

```bash
python -m pytest -q
python -m pytest --cov=fraud_detection --cov-report=term-missing
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src tests
python -m compileall -q src tests
```

Frontend checks, from `frontend/`:

```bash
npm run build
npm run check
npm run typecheck
```

The local baseline is 126 backend tests and approximately 83% branch-aware coverage as of the last local run. These are repository quality baselines, not model performance metrics.

## CI

The workflow at `.github/workflows/ci.yml` runs backend lint, formatting, typing, compilation, and coverage checks on Python 3.11 and 3.13. It runs the Astro build, Astro check, and TypeScript typecheck on Node 26. The workflow is implemented locally but is pending its first push, so this description is not a claim that it has already run remotely.

No screenshots or GIFs are committed. The reproducible local demo is the source of truth: it shows model metadata, a transaction-shaped numeric input form with synthetic examples, and the FastAPI prediction result.
