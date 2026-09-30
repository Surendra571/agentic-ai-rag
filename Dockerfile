# ==========================================
# Stage 1: Build & Dependency Installation
# ==========================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies for compiling any C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolated dependency management
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Lean Production Runtime Image
# ==========================================
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create a non-privileged system user and group (UID/GID 1001)
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 --no-create-home appuser

# Copy installed virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code, ingestion scripts, and data
COPY app/ app/
COPY scripts/ scripts/
COPY data/ data/

# Set strict file ownership to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Periodic container healthcheck against /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Launch FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
