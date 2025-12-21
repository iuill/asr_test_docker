"""
FastAPI WebSocket Server for Azure Speech-to-Text.

This module provides the HTTP and WebSocket endpoints for the
Azure Speech-to-Text service using Azure AI Speech SDK.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .audio_processor import parse_audio_message, float32_to_int16_bytes, resample_audio
from .transcription_engine import AzureSTTEngine, create_engine

logger = logging.getLogger(__name__)

# Model information (dynamically updated based on diarization setting)
MODEL_INFO = {
    "id": "azure-stt",
    "name": "Azure Speech-to-Text",
    "description": "Azure AI Speech SDK (Real-time)",
    "speed": "fast",
}


def _get_model_info(enable_diarization: bool) -> dict:
    """Generate model info based on diarization setting."""
    if enable_diarization:
        return {
            "id": "azure-stt-diarization",
            "name": "Azure Speech-to-Text (話者識別)",
            "description": "Azure AI Speech SDK (ConversationTranscriber)",
            "speed": "fast",
            "diarization_enabled": True,
        }
    else:
        return {
            "id": "azure-stt",
            "name": "Azure Speech-to-Text",
            "description": "Azure AI Speech SDK (Real-time)",
            "speed": "fast",
            "diarization_enabled": False,
        }

# Global engine instance
_engine: Optional[AzureSTTEngine] = None


def get_engine() -> AzureSTTEngine:
    """Get the global transcription engine instance."""
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _engine, MODEL_INFO

    # Startup
    logger.info("Starting up Azure STT service...")

    # Initialize transcription engine
    speech_key = getattr(app.state, "speech_key", "")
    speech_region = getattr(app.state, "speech_region", "")
    language_code = getattr(app.state, "language_code", "ja-JP")
    sample_rate = getattr(app.state, "sample_rate", 16000)
    enable_punctuation = getattr(app.state, "enable_punctuation", True)
    enable_diarization = getattr(app.state, "enable_diarization", False)

    # Update MODEL_INFO based on diarization setting
    MODEL_INFO = _get_model_info(enable_diarization)

    _engine = create_engine(
        speech_key=speech_key,
        speech_region=speech_region,
        language_code=language_code,
        sample_rate=sample_rate,
        enable_punctuation=enable_punctuation,
        enable_diarization=enable_diarization,
    )

    mode = "diarization" if enable_diarization else "standard"
    logger.info(f"Azure STT server ready (region: {speech_region}, mode: {mode})")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if _engine:
        _engine.reset()
    _engine = None


app = FastAPI(
    title="Azure Speech-to-Text ASR",
    description="Real-time speech recognition using Azure AI Speech SDK",
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
        "speech_region": engine.speech_region,
        "language_code": engine.language_code,
    }


class SettingsRequest(BaseModel):
    """Request model for settings update."""
    enable_punctuation: Optional[bool] = None


class SettingsResponse(BaseModel):
    """Response model for settings."""
    enable_punctuation: bool


@app.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current settings."""
    engine = get_engine()
    return SettingsResponse(
        enable_punctuation=engine.enable_punctuation,
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
    )
    return SettingsResponse(
        enable_punctuation=engine.enable_punctuation,
    )


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ASR using Azure Speech-to-Text.

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
                    # Include speaker ID if available (for diarization mode)
                    if result.speaker_id:
                        response["speaker_id"] = result.speaker_id
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

                        # Convert back to bytes for Azure API
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
    speech_key: str,
    speech_region: str,
    language_code: str = "ja-JP",
    sample_rate: int = 16000,
    enable_punctuation: bool = True,
    enable_diarization: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        speech_key: Azure Speech resource key
        speech_region: Azure Speech resource region
        language_code: Language code for recognition
        sample_rate: Audio sample rate
        enable_punctuation: Enable automatic punctuation
        enable_diarization: Enable speaker diarization

    Returns:
        Configured FastAPI application
    """
    app.state.speech_key = speech_key
    app.state.speech_region = speech_region
    app.state.language_code = language_code
    app.state.sample_rate = sample_rate
    app.state.enable_punctuation = enable_punctuation
    app.state.enable_diarization = enable_diarization

    return app
