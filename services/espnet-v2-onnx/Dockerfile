# ReazonSpeech Real-time ASR Server (ESPnet ONNX) - GPU Version
# Requires: reazonspeech-espnet-v2-onnx-base:gpu

FROM reazonspeech-espnet-v2-onnx-base:gpu

WORKDIR /app

# Copy application code
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install the application (--no-deps to avoid reinstalling onnxruntime CPU version)
RUN pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir fastapi uvicorn websockets "numpy>=1.24.0,<2.0.0" huggingface-hub soundfile

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "src.main", "--device", "cuda", "--host", "0.0.0.0", "--port", "8000"]
