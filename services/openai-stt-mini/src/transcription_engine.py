"""
OpenAI Speech-to-Text Transcription Engine using gpt-4o-transcribe.

Provides streaming speech recognition using OpenAI Realtime API.
"""

import asyncio
import base64
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import websockets

logger = logging.getLogger(__name__)

# OpenAI Realtime API endpoint
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


@dataclass
class TranscriptionResult:
    """Result from transcription."""

    text: str
    start_time: float
    end_time: float
    is_partial: bool
    speaker_tag: int = 0
    confidence: float = 0.0
    logprobs: Optional[list] = None


class OpenAISTTEngine:
    """
    OpenAI Speech-to-Text streaming transcription engine.

    This engine uses OpenAI Realtime API with gpt-4o-transcribe for real-time
    speech recognition with streaming support.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-transcribe",
        language: str = "ja",
        sample_rate: int = 24000,
        noise_reduction: Optional[str] = "near_field",
        include_logprobs: bool = False,
    ):
        """
        Initialize the OpenAI STT engine.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (gpt-4o-transcribe or gpt-4o-mini-transcribe)
            language: Language code for recognition (default: ja)
            sample_rate: Audio sample rate in Hz (must be 24000 for Realtime API)
            noise_reduction: Noise reduction type ("near_field", "far_field", or None)
            include_logprobs: Whether to include logprobs for confidence scores
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.noise_reduction = noise_reduction
        self.include_logprobs = include_logprobs
        self._loaded = False

        # Streaming state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._streaming_task: Optional[asyncio.Task] = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._result_queue: Optional[asyncio.Queue] = None
        self._is_streaming = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_text = ""
        self._audio_start_time = 0.0

    def load(self) -> None:
        """Validate the API key is available."""
        if self._loaded:
            return

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it to your OpenAI API key."
            )

        self._loaded = True
        logger.info(f"OpenAI STT engine initialized with model: {self.model}")

    def is_loaded(self) -> bool:
        """Check if the engine is loaded."""
        return self._loaded

    async def _connect_to_openai(self) -> None:
        """Connect to OpenAI Realtime API."""
        # Use intent=transcription for transcription-only mode
        url = f"{OPENAI_REALTIME_URL}?intent=transcription"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        logger.info(f"Connecting to OpenAI Realtime API: {url}")

        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )

        logger.info("Connected to OpenAI Realtime API")

        # Configure the session for transcription-only mode
        # Note: transcription sessions use transcription_session.update with different schema
        session_config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                    "language": self.language,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 150,
                },
            },
        }

        # Add noise reduction if configured (for transcription sessions)
        if self.noise_reduction:
            session_config["session"]["input_audio_noise_reduction"] = {
                "type": self.noise_reduction,
            }

        # Add logprobs if configured
        if self.include_logprobs:
            session_config["session"]["include"] = [
                "item.input_audio_transcription.logprobs"
            ]

        await self._ws.send(json.dumps(session_config))
        logger.info("Session configured for transcription")

    async def _send_audio_loop(self) -> None:
        """Send audio data to OpenAI in a loop."""
        try:
            while self._is_streaming:
                try:
                    audio_base64 = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.1,
                    )
                    if audio_base64 is None:
                        break

                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_base64,
                    }
                    await self._ws.send(json.dumps(message))

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")
                    break

        except Exception as e:
            logger.error(f"Audio send loop error: {e}")

    async def _receive_loop(self) -> None:
        """Receive and process messages from OpenAI."""
        try:
            async for message in self._ws:
                if not self._is_streaming:
                    break

                try:
                    data = json.loads(message)
                    event_type = data.get("type", "")

                    # Debug: log all received events
                    logger.debug(f"Received event: {event_type}, data: {json.dumps(data, ensure_ascii=False)[:500]}")

                    if event_type == "error":
                        error_msg = data.get("error", {}).get("message", "Unknown error")
                        logger.error(f"OpenAI API error: {error_msg}")
                        if self._result_queue:
                            await self._result_queue.put(
                                TranscriptionResult(
                                    text=f"[Error: {error_msg}]",
                                    start_time=0.0,
                                    end_time=0.0,
                                    is_partial=False,
                                )
                            )

                    elif event_type == "conversation.item.input_audio_transcription.delta":
                        # Partial transcription update
                        delta_text = data.get("delta", "")
                        self._current_text += delta_text
                        logger.debug(f"Delta received: '{delta_text}', current_text: '{self._current_text}'")

                        if self._result_queue and self._current_text.strip():
                            logger.info(f"Sending partial result: '{self._current_text}'")
                            await self._result_queue.put(
                                TranscriptionResult(
                                    text=self._current_text,
                                    start_time=self._audio_start_time,
                                    end_time=0.0,
                                    is_partial=True,
                                )
                            )
                        else:
                            logger.warning(f"Cannot send partial: queue={self._result_queue is not None}, text='{self._current_text}'")

                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        # Final transcription
                        final_text = data.get("transcript", self._current_text)
                        logprobs = data.get("logprobs")

                        # Calculate confidence from logprobs if available
                        confidence = 0.0
                        if logprobs:
                            try:
                                # Average of token logprobs converted to probability
                                import math
                                probs = [math.exp(lp) for lp in logprobs if lp is not None]
                                if probs:
                                    confidence = sum(probs) / len(probs)
                            except Exception as e:
                                logger.warning(f"Error calculating confidence from logprobs: {e}")

                        if self._result_queue and final_text.strip():
                            await self._result_queue.put(
                                TranscriptionResult(
                                    text=final_text,
                                    start_time=self._audio_start_time,
                                    end_time=0.0,
                                    is_partial=False,
                                    confidence=confidence,
                                    logprobs=logprobs,
                                )
                            )

                        self._current_text = ""

                    elif event_type == "input_audio_buffer.speech_started":
                        # New speech segment started
                        self._audio_start_time = data.get("audio_start_ms", 0) / 1000.0
                        self._current_text = ""

                    elif event_type == "input_audio_buffer.speech_stopped":
                        # Speech segment ended
                        pass

                    elif event_type == "session.created":
                        logger.info("Session created successfully")

                    elif event_type == "session.updated":
                        logger.info("Session updated successfully")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.ConnectionClosed as e:
            logger.info(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
        finally:
            self._is_streaming = False

    async def _streaming_loop(self) -> None:
        """Main streaming loop that handles connection and communication."""
        try:
            await self._connect_to_openai()

            # Run send and receive loops concurrently
            send_task = asyncio.create_task(self._send_audio_loop())
            receive_task = asyncio.create_task(self._receive_loop())

            await asyncio.gather(send_task, receive_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"Streaming loop error: {e}")
            if self._result_queue:
                await self._result_queue.put(
                    TranscriptionResult(
                        text=f"[Connection Error: {str(e)}]",
                        start_time=0.0,
                        end_time=0.0,
                        is_partial=False,
                    )
                )
        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None
            self._is_streaming = False

    def start_streaming(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        """
        Start streaming recognition.

        Args:
            loop: The asyncio event loop to use for results

        Returns:
            An asyncio.Queue that will receive TranscriptionResult objects
        """
        if self._is_streaming:
            raise RuntimeError("Streaming already in progress")

        self._loop = loop
        self._result_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
        self._is_streaming = True
        self._current_text = ""
        self._audio_start_time = 0.0

        # Start the streaming task
        self._streaming_task = asyncio.create_task(self._streaming_loop())

        return self._result_queue

    async def add_audio_async(self, audio_base64: str) -> None:
        """
        Add audio data to the streaming queue (async version).

        Args:
            audio_base64: Base64-encoded audio data (16-bit PCM, 24kHz)
        """
        if self._is_streaming:
            await self._audio_queue.put(audio_base64)

    def add_audio(self, audio_base64: str) -> None:
        """
        Add audio data to the streaming queue.

        Args:
            audio_base64: Base64-encoded audio data (16-bit PCM, 24kHz)
        """
        if self._is_streaming and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._audio_queue.put(audio_base64),
                self._loop,
            )

    async def stop_streaming_async(self) -> None:
        """Stop streaming recognition (async version)."""
        if not self._is_streaming:
            return

        self._is_streaming = False

        # Signal end of audio
        await self._audio_queue.put(None)

        # Commit the audio buffer
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            except Exception as e:
                logger.warning(f"Error committing audio buffer: {e}")

        # Wait for streaming task to finish
        if self._streaming_task:
            try:
                await asyncio.wait_for(self._streaming_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._streaming_task.cancel()
            except Exception as e:
                logger.warning(f"Error waiting for streaming task: {e}")

        self._streaming_task = None
        self._result_queue = None

    def stop_streaming(self) -> None:
        """Stop streaming recognition."""
        if not self._is_streaming:
            return

        if self._loop:
            future = asyncio.run_coroutine_threadsafe(
                self.stop_streaming_async(),
                self._loop,
            )
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"Error stopping streaming: {e}")

    def reset(self) -> None:
        """Reset the engine state."""
        self.stop_streaming()
        self._current_text = ""
        self._audio_start_time = 0.0

    def update_settings(
        self,
        language: Optional[str] = None,
        model: Optional[str] = None,
        noise_reduction: Optional[str] = None,
        include_logprobs: Optional[bool] = None,
    ) -> None:
        """
        Update engine settings.

        Args:
            language: Language code for recognition
            model: Model to use
            noise_reduction: Noise reduction type ("near_field", "far_field", or None)
            include_logprobs: Whether to include logprobs for confidence scores
        """
        if language is not None:
            self.language = language
        if model is not None:
            self.model = model
        if noise_reduction is not None:
            self.noise_reduction = noise_reduction
        if include_logprobs is not None:
            self.include_logprobs = include_logprobs

        logger.info(
            f"Settings updated: language={self.language}, model={self.model}, "
            f"noise_reduction={self.noise_reduction}, include_logprobs={self.include_logprobs}"
        )


def create_engine(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-transcribe",
    language: str = "ja",
    sample_rate: int = 24000,
    noise_reduction: Optional[str] = "near_field",
    include_logprobs: bool = False,
) -> OpenAISTTEngine:
    """
    Create and initialize an OpenAI STT engine.

    Args:
        api_key: OpenAI API key
        model: Model to use
        language: Language code for recognition
        sample_rate: Audio sample rate in Hz
        noise_reduction: Noise reduction type ("near_field", "far_field", or None)
        include_logprobs: Whether to include logprobs for confidence scores

    Returns:
        Initialized OpenAISTTEngine instance
    """
    engine = OpenAISTTEngine(
        api_key=api_key,
        model=model,
        language=language,
        sample_rate=sample_rate,
        noise_reduction=noise_reduction,
        include_logprobs=include_logprobs,
    )
    engine.load()
    return engine
