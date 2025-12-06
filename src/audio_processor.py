"""
Audio Processor for real-time speech recognition.

This module handles the audio processing pipeline:
1. Receive audio chunks from WebSocket
2. Apply VAD to detect speech segments
3. Send speech segments to transcription engine
4. Return transcription results
"""

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Optional

import numpy as np

from .transcription_engine import TranscriptionEngine, TranscriptionResult
from .vad import AudioChunker, SileroVAD

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class ProcessingResult:
    """Result from audio processing."""

    text: str
    start_time: float
    end_time: float
    is_partial: bool = False


class AudioProcessor:
    """
    Processes audio streams for real-time speech recognition.

    This class coordinates VAD, chunking, and transcription to provide
    pseudo-real-time speech recognition using an offline model.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        vad_threshold: float = 0.5,
        max_chunk_duration: float = 5.0,
        min_chunk_duration: float = 0.5,
    ):
        """
        Initialize the audio processor.

        Args:
            engine: Transcription engine instance
            vad_threshold: VAD speech probability threshold
            max_chunk_duration: Maximum chunk duration in seconds
            min_chunk_duration: Minimum chunk duration in seconds
        """
        self.engine = engine

        # Initialize VAD
        self.vad = SileroVAD(threshold=vad_threshold)
        self.vad.load_model()

        # Initialize chunker
        self.chunker = AudioChunker(
            vad=self.vad,
            max_chunk_duration=max_chunk_duration,
            min_chunk_duration=min_chunk_duration,
        )

        self._processing = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._result_callback: Optional[Callable[[ProcessingResult], None]] = None

    def set_result_callback(self, callback: Callable[[ProcessingResult], None]):
        """Set callback function for transcription results."""
        self._result_callback = callback

    async def process_audio_bytes(self, audio_bytes: bytes) -> list[ProcessingResult]:
        """
        Process raw audio bytes (16-bit PCM, mono, 16kHz).

        Args:
            audio_bytes: Raw audio data

        Returns:
            List of processing results
        """
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        return await self.process_audio(audio)

    async def process_audio(self, audio: np.ndarray) -> list[ProcessingResult]:
        """
        Process audio data and return transcription results.

        Args:
            audio: Audio data as numpy array (float32, mono)

        Returns:
            List of processing results
        """
        results = []

        # Get speech segments from chunker
        segments = self.chunker.add_audio(audio)

        # Transcribe each segment
        for segment in segments:
            try:
                transcription = self.engine.transcribe(segment.audio)

                if transcription.text:
                    result = ProcessingResult(
                        text=transcription.text,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        is_partial=False,
                    )
                    results.append(result)

                    if self._result_callback:
                        self._result_callback(result)

            except Exception as e:
                logger.error(f"Transcription error: {e}")

        return results

    async def flush(self) -> Optional[ProcessingResult]:
        """
        Flush any remaining audio and return final result.

        Returns:
            Final processing result if any
        """
        segment = self.chunker.flush()

        if segment is not None:
            try:
                transcription = self.engine.transcribe(segment.audio)

                if transcription.text:
                    result = ProcessingResult(
                        text=transcription.text,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        is_partial=False,
                    )

                    if self._result_callback:
                        self._result_callback(result)

                    return result

            except Exception as e:
                logger.error(f"Transcription error during flush: {e}")

        return None

    def reset(self):
        """Reset the processor state for a new audio stream."""
        self.chunker.reset()


class AudioStreamHandler:
    """
    Handles WebSocket audio streams.

    This class manages the connection lifecycle and audio processing
    for a single WebSocket client.
    """

    def __init__(self, processor: AudioProcessor):
        """
        Initialize the stream handler.

        Args:
            processor: Audio processor instance
        """
        self.processor = processor
        self._active = False

    async def handle_stream(
        self, audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[ProcessingResult, None]:
        """
        Handle an audio stream and yield transcription results.

        Args:
            audio_stream: Async generator yielding audio bytes

        Yields:
            ProcessingResult objects as transcription completes
        """
        self._active = True
        self.processor.reset()

        try:
            async for audio_bytes in audio_stream:
                if not self._active:
                    break

                results = await self.processor.process_audio_bytes(audio_bytes)
                for result in results:
                    yield result

            # Flush remaining audio
            final_result = await self.processor.flush()
            if final_result:
                yield final_result

        finally:
            self._active = False
            self.processor.reset()

    def stop(self):
        """Stop processing the current stream."""
        self._active = False


def parse_audio_message(message: bytes) -> tuple[int, np.ndarray]:
    """
    Parse audio message from WebSocket.

    Expected format:
    - First 4 bytes: sample rate (little-endian int32)
    - Remaining bytes: 16-bit PCM audio data

    Args:
        message: Raw message bytes

    Returns:
        Tuple of (sample_rate, audio_array)
    """
    if len(message) < 4:
        raise ValueError("Message too short")

    sample_rate = struct.unpack("<I", message[:4])[0]
    audio_bytes = message[4:]

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    return sample_rate, audio
