"""
FastAPI WebSocket Server for OpenAI Speech-to-Text.

This module provides the HTTP and WebSocket endpoints for the
OpenAI Speech-to-Text service using gpt-4o-transcribe.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .audio_processor import (
    parse_audio_message,
    float32_to_int16_bytes,
    int16_bytes_to_base64,
    resample_audio,
)
from .transcription_engine import OpenAISTTEngine, create_engine

logger = logging.getLogger(__name__)

# Model information
MODEL_INFO = {
    "id": "openai-stt",
    "name": "OpenAI gpt-4o-transcribe",
    "description": "OpenAI Realtime API (gpt-4o-transcribe)",
    "speed": "fast",
}

# Global engine instance
_engine: Optional[OpenAISTTEngine] = None


def get_engine() -> OpenAISTTEngine:
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
    logger.info("Starting up OpenAI STT service...")

    # Initialize transcription engine
    model = getattr(app.state, "model", "gpt-4o-transcribe")
    language = getattr(app.state, "language", "ja")
    sample_rate = getattr(app.state, "sample_rate", 24000)

    _engine = create_engine(
        model=model,
        language=language,
        sample_rate=sample_rate,
    )

    logger.info("OpenAI STT server ready")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if _engine:
        _engine.reset()
    _engine = None


app = FastAPI(
    title="OpenAI Speech-to-Text ASR",
    description="Real-time speech recognition using OpenAI gpt-4o-transcribe",
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


class SettingsRequest(BaseModel):
    """Request model for settings update."""
    language: Optional[str] = None
    model: Optional[str] = None


class SettingsResponse(BaseModel):
    """Response model for settings."""
    language: str
    model: str


@app.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current settings."""
    engine = get_engine()
    return SettingsResponse(
        language=engine.language,
        model=engine.model,
    )


@app.post("/settings", response_model=SettingsResponse)
async def update_settings(request: SettingsRequest):
    """
    Update recognition settings.

    Note: Changes take effect on the next streaming session.
    """
    engine = get_engine()
    engine.update_settings(
        language=request.language,
        model=request.model,
    )
    return SettingsResponse(
        language=engine.language,
        model=engine.model,
    )


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASR using OpenAI gpt-4o-transcribe.

    Protocol:
    - Client sends binary audio data (16-bit PCM, mono)
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
                    logger.info(f"Got result from queue: text='{result.text}', is_partial={result.is_partial}")
                    response = {
                        "type": "transcription",
                        "text": result.text,
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "is_final": not result.is_partial,
                        "provider_info": {
                            "confidence": result.confidence,
                        },
                    }
                    if result.speaker_tag > 0:
                        response["speaker_tag"] = result.speaker_tag
                    logger.info(f"Sending to WebSocket: {response}")
                    await websocket.send_json(response)
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

                        # Resample to 24kHz if needed (OpenAI Realtime API requirement)
                        if sample_rate != 24000:
                            audio = resample_audio(audio, sample_rate, 24000)

                        # Convert to 16-bit PCM bytes
                        audio_bytes_24k = float32_to_int16_bytes(audio)

                        # Encode to base64 for OpenAI API
                        audio_base64 = int16_bytes_to_base64(audio_bytes_24k)

                        # Add to streaming queue
                        await engine.add_audio_async(audio_base64)

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
                        await engine.stop_streaming_async()
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
    model: str = "gpt-4o-transcribe",
    language: str = "ja",
    sample_rate: int = 24000,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        model: Model to use
        language: Language code for recognition
        sample_rate: Audio sample rate

    Returns:
        Configured FastAPI application
    """
    app.state.model = model
    app.state.language = language
    app.state.sample_rate = sample_rate

    return app
