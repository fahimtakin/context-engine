# ==========================================
# STAGE 1: Build & Package Compilation
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system compilation packages and git natively inside the build sandbox
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements mapping sheet early to optimize Docker layer caching
COPY requirements.txt .

# Upgrade pip and compile packages globally inside the build boundary stage
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ==========================================
# STAGE 2: Slim Production Runtime Runner
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install git inside the runner environment so GitPython can execute commands natively
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a secure, isolated non-root system process execution user
RUN useradd -u 8888 appuser && mkdir -p /app/documents && chown -R appuser:appuser /app
USER appuser

# Copy pristine pre-compiled packages directly into the production layer path
COPY --from=builder /install /usr/local
COPY --from=builder /build/requirements.txt .

# Copy your core custom application logic modules into the container layout scope
COPY --chown=appuser:appuser app/ ./app
COPY --chown=appuser:appuser run_api.py .
COPY --chown=appuser:appuser run_ingestion.py .

# Copy environment layout configuration sheet natively to the app sandbox layer
COPY --chown=appuser:appuser .env . 

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Execute using standard string module format to preserve multi-worker thread spawning
CMD ["python", "run_api.py"]
