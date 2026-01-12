# ============================================
# Luminark Backend - Hugging Face Spaces
# CPU-optimized Docker deployment
# ============================================

FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements/base.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# PyTorch CPU-only
RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Application code
COPY core/ /app/core/
COPY backend/ /app/backend/
COPY models/ /app/models/

# Environment
ENV PYTHONPATH=/app
ENV MODEL_DEVICE=cpu
ENV LUMINARK_API_KEYS=lum_prod_key_secure_2026

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run on HF Spaces port
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
