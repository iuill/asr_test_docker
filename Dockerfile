# ReazonSpeech Real-time ASR Server - GPU Version
# Base image with CUDA support

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (for Silero VAD)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install sherpa-onnx with CUDA support
RUN pip install --no-cache-dir sherpa-onnx

# Install other dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    websockets>=12.0 \
    numpy>=1.24.0 \
    huggingface-hub>=0.19.0

# Copy application code
COPY src/ /app/src/
COPY pyproject.toml /app/

# Install the application
RUN pip install --no-cache-dir -e .

# Pre-download the model (optional, can be done at runtime)
# RUN python -c "from src.transcription_engine import TranscriptionEngine; e = TranscriptionEngine(); e._download_model()"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "src.main", "--device", "cuda", "--host", "0.0.0.0", "--port", "8000"]
