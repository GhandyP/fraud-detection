"""Run the fraud detection API with ``python -m fraud_detection.api``."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from fraud_detection.api.app import create_app


def main() -> None:
    """Start uvicorn using environment-configured server settings."""
    root = Path(os.environ.get("FRAUD_API_ROOT", Path.cwd()))
    host = os.environ.get("FRAUD_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FRAUD_API_PORT", "8000"))
    uvicorn.run(create_app(root=root), host=host, port=port)


if __name__ == "__main__":
    main()
