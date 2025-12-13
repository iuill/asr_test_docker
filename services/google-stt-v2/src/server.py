"""
FastAPI WebSocket Server for Google Speech-to-Text V2.

This module provides the HTTP and WebSocket endpoints for the
Google Speech-to-Text V2 service with Chirp 2/3 models.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .audio_processor import parse_audio_message, float32_to_int16_bytes, resample_audio
from .transcription_engine import GoogleSTTV2Engine, create_engine

logger = logging.getLogger(__name__)

# Model information
MODEL_INFO = {
    "id": "google-stt-v2",
    "name": "Google Speech-to-Text V2",
    "description": "Google Cloud Speech-to-Text V2 API (Chirp 2/3)",
    "speed": "fast",
}

# Global engine instance
_engine: Optional[GoogleSTTV2Engine] = None


def get_engine() -> GoogleSTTV2Engine:
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
    logger.info("Starting up Google STT V2 service...")

    # Initialize transcription engine
    project_id = getattr(app.state, "project_id", "")
    location = getattr(app.state, "location", "us")
    language_code = getattr(app.state, "language_code", "ja-JP")
    sample_rate = getattr(app.state, "sample_rate", 16000)
    enable_punctuation = getattr(app.state, "enable_punctuation", True)
    enable_diarization = getattr(app.state, "enable_diarization", False)
    diarization_speaker_count = getattr(app.state, "diarization_speaker_count", 2)
    model = getattr(app.state, "model", "chirp_2")

    _engine = create_engine(
        project_id=project_id,
        location=location,
        language_code=language_code,
        sample_rate=sample_rate,
        enable_punctuation=enable_punctuation,
        enable_diarization=enable_diarization,
        diarization_speaker_count=diarization_speaker_count,
        model=model,
    )

    logger.info(f"Google STT V2 server ready (model: {model})")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if _engine:
        _engine.reset()
    _engine = None


app = FastAPI(
    title="Google Speech-to-Text V2 ASR",
    description="Real-time speech recognition using Google Cloud Speech-to-Text V2 (Chirp 2/3)",
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
        "recognition_model": engine.model,
    }


class SettingsRequest(BaseModel):
    """Request model for settings update."""
    enable_punctuation: Optional[bool] = None
    enable_diarization: Optional[bool] = None
    diarization_speaker_count: Optional[int] = None


class SettingsResponse(BaseModel):
    """Response model for settings."""
    enable_punctuation: bool
    enable_diarization: bool
    diarization_speaker_count: int


@app.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current settings."""
    engine = get_engine()
    return SettingsResponse(
        enable_punctuation=engine.enable_punctuation,
        enable_diarization=engine.enable_diarization,
        diarization_speaker_count=engine.diarization_speaker_count,
    )


@app.post("/settings", response_model=SettingsResponse)
async def update_settings(request: SettingsRequest):
    """
    Update recognition settings.

    Note: Changes take effect on the next streaming session.
    """
    engine = get_engine()
    engine.update_settings(
        enable_punctuation=request.enable_punctuation,
        enable_diarization=request.enable_diarization,
        diarization_speaker_count=request.diarization_speaker_count,
    )
    return SettingsResponse(
        enable_punctuation=engine.enable_punctuation,
        enable_diarization=engine.enable_diarization,
        diarization_speaker_count=engine.diarization_speaker_count,
    )


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASR using Google Speech-to-Text V2.

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
                    response = {
                        "type": "transcription",
                        "text": result.text,
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "is_final": not result.is_partial,
                        # Provider-specific info for debugging
                        "provider_info": {
                            "stability": result.stability,
                            "confidence": result.confidence,
                            "result_index": result.result_index,
                        },
                    }
                    # Include speaker tag if diarization is enabled
                    if result.speaker_tag > 0:
                        response["speaker_tag"] = result.speaker_tag
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
    project_id: str,
    location: str = "us",
    language_code: str = "ja-JP",
    sample_rate: int = 16000,
    enable_punctuation: bool = True,
    enable_diarization: bool = False,
    diarization_speaker_count: int = 2,
    model: str = "chirp_2",
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        project_id: Google Cloud project ID
        location: API location (e.g., "us", "eu", "us-central1")
        language_code: Language code for recognition
        sample_rate: Audio sample rate
        enable_punctuation: Enable automatic punctuation
        enable_diarization: Enable speaker diarization
        diarization_speaker_count: Expected number of speakers
        model: Recognition model to use ("chirp_2", "chirp_3")

    Returns:
        Configured FastAPI application
    """
    app.state.project_id = project_id
    app.state.location = location
    app.state.language_code = language_code
    app.state.sample_rate = sample_rate
    app.state.enable_punctuation = enable_punctuation
    app.state.enable_diarization = enable_diarization
    app.state.diarization_speaker_count = diarization_speaker_count
    app.state.model = model

    return app
