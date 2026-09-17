FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_API_HOST=0.0.0.0 \
    FRAUD_API_PORT=8000

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin app

COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Runtime configuration is required to resolve the mounted model artifact.
COPY config/ ./config/

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["python", "-m", "fraud_detection.api"]
