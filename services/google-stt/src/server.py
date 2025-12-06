"""
FastAPI WebSocket Server for Google Speech-to-Text.

This module provides the HTTP and WebSocket endpoints for the
Google Speech-to-Text service.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .audio_processor import parse_audio_message, float32_to_int16_bytes, resample_audio
from .transcription_engine import GoogleSTTEngine, create_engine

logger = logging.getLogger(__name__)

# Model information
MODEL_INFO = {
    "id": "google-stt",
    "name": "Google Speech-to-Text",
    "description": "Google Cloud Speech-to-Text API (Streaming)",
    "speed": "fast",
}

# Global engine instance
_engine: Optional[GoogleSTTEngine] = None


def get_engine() -> GoogleSTTEngine:
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
    logger.info("Starting up Google STT service...")

    # Initialize transcription engine
    language_code = getattr(app.state, "language_code", "ja-JP")
    sample_rate = getattr(app.state, "sample_rate", 16000)

    _engine = create_engine(
        language_code=language_code,
        sample_rate=sample_rate,
    )

    logger.info("Google STT server ready")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if _engine:
        _engine.reset()
    _engine = None


app = FastAPI(
    title="Google Speech-to-Text ASR",
    description="Real-time speech recognition using Google Cloud Speech-to-Text",
    version="0.1.0",
    lifespan=lifespan,
)


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
    WebSocket endpoint for real-time ASR using Google Speech-to-Text.

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

    engine = get_engine()
    result_queue: Optional[asyncio.Queue] = None

    try:
        # Start streaming recognition
        loop = asyncio.get_event_loop()
        result_queue = engine.start_streaming(loop)

        # Task to send results to client
        async def send_results():
            try:
                while True:
                    result = await result_queue.get()
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result.text,
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "is_final": not result.is_partial,
                    })
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error sending results: {e}")

        # Start result sender task
        result_task = asyncio.create_task(send_results())

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
                            audio = resample_audio(audio, sample_rate, 16000)

                        # Convert back to bytes for Google API
                        audio_bytes_16k = float32_to_int16_bytes(audio)

                        # Add to streaming queue
                        engine.add_audio(audio_bytes_16k)

                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })

                elif "text" in message:
                    # Control message
                    text = message["text"]
                    data = json.loads(text)

                    if data.get("type") == "end":
                        # Client signals end of audio
                        engine.stop_streaming()
                        await websocket.send_json({"type": "end"})
                        break

                    elif data.get("type") == "reset":
                        # Reset engine state
                        engine.reset()
                        # Restart streaming
                        result_queue = engine.start_streaming(loop)
                        await websocket.send_json({"type": "reset_ack"})

        finally:
            # Cancel result sender
            result_task.cancel()
            try:
                await result_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        engine.reset()


def create_app(
    language_code: str = "ja-JP",
    sample_rate: int = 16000,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        language_code: Language code for recognition
        sample_rate: Audio sample rate

    Returns:
        Configured FastAPI application
    """
    app.state.language_code = language_code
    app.state.sample_rate = sample_rate

    return app
