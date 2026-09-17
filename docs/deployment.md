# Local deployment

This optional Docker Compose setup runs the FastAPI inference service and the static Astro frontend. It does not train models.

## Prerequisites

- Docker with Docker Compose support
- A trained model artifact mounted from `models/trained/`

Produce the artifact with the local workflow, outside the containers:

```bash
fraud train --json
```

## Build and run

```bash
docker compose build
docker compose up
```

The services are available at:

| Service | Address | Container port |
| --- | --- | --- |
| Backend API | http://localhost:8000 | 8000 |
| Frontend | http://localhost:4321 | 80 |

The frontend image receives `PUBLIC_API_BASE_URL` as a build argument. Runtime backend settings are configured with these variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `FRAUD_API_ROOT` | Backend project root | `/app` |
| `FRAUD_API_HOST` | Backend bind host | `0.0.0.0` |
| `FRAUD_API_PORT` | Backend listen port | `8000` |
| `FRAUD_API_MAX_BODY_BYTES` | Maximum request body size | `32768` |
| `FRAUD_API_CORS_ORIGINS` | Comma-separated browser origins | `http://localhost:4321,http://127.0.0.1:4321` |
| `BACKEND_PORT` | Host port mapped to backend | `8000` |
| `PUBLIC_API_BASE_URL` | Frontend build-time API URL | `http://localhost:8000` |

The backend health check calls `/health`; Compose starts the frontend only after the backend is healthy. A failing backend health check commonly means the trained artifact is missing from `models/trained/` (or cannot be loaded), so the frontend is not started.

The trust boundary is unchanged: the browser never receives the trained artifact. Only the FastAPI API loads and executes the model.

## Verification and teardown

This repository checkout did not build images because the Docker socket is root-only in this environment. A user with Docker access should verify the setup with:

```bash
docker compose build
docker compose up
curl http://localhost:8000/health
# Open http://localhost:4321 in a browser.
```

Stop and remove the Compose services with:

```bash
docker compose down
```
