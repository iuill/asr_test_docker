# ReazonSpeech Real-time ASR Server - GPU Version
# Requires: reazonspeech-k2-v2-base:gpu

FROM reazonspeech-k2-v2-base:gpu

WORKDIR /app

# Copy application code
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install the application
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "src.main", "--device", "cuda", "--host", "0.0.0.0", "--port", "8000"]
