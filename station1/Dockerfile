# ═══════════════════════════════════════════════════════════════════════════════
# Station 1 — ESM-2 Embedding Service (CPU-only Docker build)
#
# Multi-stage build:
#   Stage 1 — install Python dependencies (cached layer)
#   Stage 2 — copy application code (fast rebuild on code change)
#
# PyTorch is installed from the CPU-only wheel index to avoid shipping ~2 GB
# of CUDA libraries inside the container. If you need GPU inference, run
# natively (see README.md Step 2) or build on an nvidia/cuda base image.
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS deps

WORKDIR /app

# System deps for potential native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST (avoids pulling CUDA wheels)
RUN pip install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps (torch already satisfied, pip will skip it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY embed.py main.py ./

# Create output directory for .npy files
RUN mkdir -p /app/outputs

# The model is downloaded on first startup (~2.5 GB) and cached inside the
# container at /root/.cache/huggingface. To persist across restarts, mount
# a volume: -v huggingface_cache:/root/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
