# syntax=docker/dockerfile:1
# ReefScan AI — dashboard + CPU inference image (~2.5 GB with torch-CPU wheels).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    TORCH_HOME=/app/.torch \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# libglib2.0-0: runtime shared lib OpenCV-headless dlopens (VideoWriter/imread).
# git omitted on purpose: .git is not in the image, so provenance.git_sha()
# returns "unknown"; pass a build-arg if you need the SHA stamped in manifests.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Layer 1: dependencies (cached until requirements/constraints change) ----
COPY requirements.txt constraints.txt ./
# CPU wheel index FIRST (PyPI linux default pulls ~2.5 GB of CUDA deps),
# then the project — torch is already satisfied and is not re-resolved.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -c constraints.txt \
 && pip install -r requirements.txt -c constraints.txt

# ---- Layer 2: application code (no tests/docs/notebooks/data in the image) ----
COPY app.py ./
COPY src/ ./src/
COPY configs/ ./configs/

# Bind-mount points; .torch cache dir must be writable by the app user.
RUN mkdir -p models results reports datasets .torch \
 && useradd --create-home --shell /bin/bash reefscan \
 && chown -R reefscan:reefscan /app
USER reefscan

EXPOSE 8501

# Streamlit's built-in health endpoint returns 200 "ok" once the server is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stalker/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
