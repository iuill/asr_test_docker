"""
FastAPI WebSocket Server for real-time speech recognition.

This module provides the HTTP and WebSocket endpoints for the
speech recognition service using espnet-v2.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .audio_processor import AudioProcessor, parse_audio_message
from .transcription_engine import TranscriptionEngine, create_engine

logger = logging.getLogger(__name__)

# Model information
MODEL_INFO = {
    "id": "espnet-v2",
    "name": "ESPnet-v2",
    "description": "ESPnet Conformer-Transducer (119M params)",
    "speed": "standard",
}

# Global engine instance
_engine: Optional[TranscriptionEngine] = None


def get_engine() -> TranscriptionEngine:
    """Get the global transcription engine instance."""
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _engine

    # Startup
    logger.info("Starting up...")

    # Initialize transcription engine
    device = app.state.device if hasattr(app.state, "device") else "cpu"
    num_threads = app.state.num_threads if hasattr(app.state, "num_threads") else 4

    _engine = create_engine(device=device, num_threads=num_threads)

    logger.info("Server ready")

    yield

    # Shutdown
    logger.info("Shutting down...")
    _engine = None


app = FastAPI(
    title="Real-time ASR (ESPnet-v2)",
    description="Real-time speech recognition using espnet-v2 model",
    version="0.1.0",
    lifespan=lifespan,
)

# Track active connections
active_connections: list[WebSocket] = []


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")


manager = ConnectionManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    engine = get_engine()
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded(),
        "model": MODEL_INFO["id"],
    }


@app.get("/info")
async def get_info():
    """Get model information."""
    engine = get_engine()
    return {
        **MODEL_INFO,
        "model_loaded": engine.is_loaded(),
    }


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASR.

    Protocol:
    - Client sends binary audio data (16-bit PCM, mono, 16kHz)
    - Server sends JSON messages with transcription results

    Message format from client:
    - First 4 bytes: sample rate (little-endian int32)
    - Remaining bytes: audio data

    Message format to client:
    {
        "type": "transcription",
        "text": "recognized text",
        "start_time": 0.0,
        "end_time": 1.5,
        "is_final": true
    }
    """
    await manager.connect(websocket)

    # Create processor for this connection
    engine = get_engine()
    processor = AudioProcessor(engine)

    try:
        while True:
            # Receive audio data
            message = await websocket.receive()

            if "bytes" in message:
                # Binary audio data
                audio_bytes = message["bytes"]

                try:
                    sample_rate, audio = parse_audio_message(audio_bytes)

                    # Resample if needed
                    if sample_rate != 16000:
                        # Simple resampling
                        duration = len(audio) / sample_rate
                        new_length = int(duration * 16000)
                        indices = np.linspace(0, len(audio) - 1, new_length)
                        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

                    # Process audio
                    results = await processor.process_audio(audio)

                    # Send results
                    for result in results:
                        await websocket.send_json(
                            {
                                "type": "transcription",
                                "text": result.text,
                                "start_time": result.start_time,
                                "end_time": result.end_time,
                                "is_final": not result.is_partial,
                            }
                        )

                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif "text" in message:
                # Control message
                text = message["text"]
                data = json.loads(text)

                if data.get("type") == "end":
                    # Client signals end of audio
                    final_result = await processor.flush()
                    if final_result:
                        await websocket.send_json(
                            {
                                "type": "transcription",
                                "text": final_result.text,
                                "start_time": final_result.start_time,
                                "end_time": final_result.end_time,
                                "is_final": True,
                            }
                        )

                    await websocket.send_json({"type": "end"})
                    break

                elif data.get("type") == "reset":
                    # Reset processor state
                    processor.reset()
                    await websocket.send_json({"type": "reset_ack"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        processor.reset()


def create_app(device: str = "cpu", num_threads: int = 4) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        device: Device to use ('cpu' or 'cuda')
        num_threads: Number of threads for inference

    Returns:
        Configured FastAPI application
    """
    app.state.device = device
    app.state.num_threads = num_threads

    return app
