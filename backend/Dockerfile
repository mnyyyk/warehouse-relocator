# Backend Dockerfile for FastAPI (Uvicorn) on AWS App Runner

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (minimal). psycopg2-binary is used so libpq headers are not necessary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
  && pip install -r /app/requirements.txt

# Copy backend source
COPY . /app

# Expose port (App Runner injects PORT env, default 8000)
ENV PORT=8000

# Healthcheck path
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Use entrypoint script: run Alembic migrations then start uvicorn
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
