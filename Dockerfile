# ---------- Stage 1: build wheels ----------
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: slim runtime ----------
FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# git is needed at runtime to clone repositories for indexing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 8888 --create-home appuser

COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser app/ ./app
COPY --chown=appuser:appuser run_api.py run_ingestion.py ./
RUN mkdir -p /app/documents && chown appuser:appuser /app/documents

# Configuration comes from environment variables (see docker-compose.yml);
# secrets are never baked into the image.
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "run_api.py"]
