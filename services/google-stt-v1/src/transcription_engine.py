"""
Google Speech-to-Text Transcription Engine.

Provides streaming speech recognition using Google Cloud Speech-to-Text API.
"""

import asyncio
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Generator, Optional

from google.cloud import speech

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from transcription."""

    text: str
    start_time: float
    end_time: float
    is_partial: bool
    speaker_tag: int = 0  # Speaker tag for diarization (0 = unknown)
    # Provider-specific info for debugging
    stability: float = 0.0  # Stability of interim results (0.0-1.0)
    confidence: float = 0.0  # Recognition confidence
    result_index: int = 0  # Index in response.results array


class GoogleSTTEngine:
    """
    Google Speech-to-Text streaming transcription engine.

    This engine uses Google Cloud Speech-to-Text API for real-time
    speech recognition with streaming support.
    """

    def __init__(
        self,
        language_code: str = "ja-JP",
        sample_rate: int = 16000,
        enable_punctuation: bool = True,
        enable_diarization: bool = False,
        diarization_speaker_count: int = 2,
    ):
        """
        Initialize the Google STT engine.

        Args:
            language_code: Language code for recognition (default: ja-JP)
            sample_rate: Audio sample rate in Hz
            enable_punctuation: Enable automatic punctuation
            enable_diarization: Enable speaker diarization
            diarization_speaker_count: Expected number of speakers (for diarization)
        """
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.enable_punctuation = enable_punctuation
        self.enable_diarization = enable_diarization
        self.diarization_speaker_count = diarization_speaker_count
        self._client: Optional[speech.SpeechClient] = None
        self._loaded = False

        # Streaming state
        self._audio_queue: queue.Queue = queue.Queue()
        self._streaming_thread: Optional[threading.Thread] = None
        self._result_queue: asyncio.Queue = None
        self._is_streaming = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def load(self) -> None:
        """Load the Google Speech client."""
        if self._loaded:
            return

        try:
            # Check for credentials
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not creds_path:
                logger.warning(
                    "GOOGLE_APPLICATION_CREDENTIALS not set. "
                    "Make sure credentials are configured."
                )

            self._client = speech.SpeechClient()
            self._loaded = True
            logger.info("Google Speech-to-Text client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Google STT client: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if the engine is loaded."""
        return self._loaded

    def get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get the streaming recognition configuration."""
        # Base recognition config
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=self.enable_punctuation,
            model="default",
        )

        # Add diarization config if enabled
        if self.enable_diarization:
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=1,
                max_speaker_count=self.diarization_speaker_count,
            )
            recognition_config.diarization_config = diarization_config

        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
        )

        return streaming_config

    def _audio_generator(self) -> Generator[speech.StreamingRecognizeRequest, None, None]:
        """Generate audio requests for streaming."""
        while self._is_streaming:
            try:
                # Get audio chunk with timeout
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None:
                    # End signal
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def _streaming_recognize_thread(self) -> None:
        """Run streaming recognition in a separate thread."""
        try:
            streaming_config = self.get_streaming_config()

            # Create the streaming request generator
            requests = self._audio_generator()

            # Start streaming recognition
            responses = self._client.streaming_recognize(
                config=streaming_config,
                requests=requests,
            )

            # Process responses
            for response in responses:
                if not self._is_streaming:
                    break

                for result_index, result in enumerate(response.results):
                    if not result.alternatives:
                        continue

                    alternative = result.alternatives[0]
                    text = alternative.transcript
                    speaker_tag = 0

                    # Extract stability and confidence for debugging
                    stability = getattr(result, 'stability', 0.0)
                    confidence = getattr(alternative, 'confidence', 0.0)

                    # Extract result_end_time
                    result_end_time = getattr(result, 'result_end_time', None)
                    if result_end_time:
                        end_time = result_end_time.seconds + result_end_time.microseconds / 1_000_000
                    else:
                        end_time = 0.0

                    # Extract speaker tag if diarization is enabled
                    if self.enable_diarization and result.is_final:
                        # Get the last word's speaker tag (most accurate for final results)
                        if alternative.words:
                            speaker_tag = alternative.words[-1].speaker_tag

                    if text.strip():
                        transcription_result = TranscriptionResult(
                            text=text,
                            start_time=0.0,
                            end_time=end_time,
                            is_partial=not result.is_final,
                            speaker_tag=speaker_tag,
                            stability=stability,
                            confidence=confidence,
                            result_index=result_index,
                        )

                        # Put result in the async queue
                        if self._loop and self._result_queue:
                            asyncio.run_coroutine_threadsafe(
                                self._result_queue.put(transcription_result),
                                self._loop,
                            )

        except Exception as e:
            logger.error(f"Streaming recognition error: {e}")
            # Signal error to the result queue
            if self._loop and self._result_queue:
                error_result = TranscriptionResult(
                    text=f"[Error: {str(e)}]",
                    start_time=0.0,
                    end_time=0.0,
                    is_partial=False,
                )
                asyncio.run_coroutine_threadsafe(
                    self._result_queue.put(error_result),
                    self._loop,
                )
        finally:
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
        self._audio_queue = queue.Queue()
        self._is_streaming = True

        # Start the streaming thread
        self._streaming_thread = threading.Thread(
            target=self._streaming_recognize_thread,
            daemon=True,
        )
        self._streaming_thread.start()

        return self._result_queue

    def add_audio(self, audio_bytes: bytes) -> None:
        """
        Add audio data to the streaming queue.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM)
        """
        if self._is_streaming:
            self._audio_queue.put(audio_bytes)

    def stop_streaming(self) -> None:
        """Stop streaming recognition."""
        if not self._is_streaming:
            return

        self._is_streaming = False

        # Signal end of audio
        self._audio_queue.put(None)

        # Wait for thread to finish
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=5.0)

        self._streaming_thread = None
        self._result_queue = None
        self._loop = None

    def reset(self) -> None:
        """Reset the engine state."""
        self.stop_streaming()
        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def update_settings(
        self,
        enable_punctuation: Optional[bool] = None,
        enable_diarization: Optional[bool] = None,
        diarization_speaker_count: Optional[int] = None,
    ) -> None:
        """
        Update engine settings.

        Args:
            enable_punctuation: Enable automatic punctuation
            enable_diarization: Enable speaker diarization
            diarization_speaker_count: Expected number of speakers
        """
        if enable_punctuation is not None:
            self.enable_punctuation = enable_punctuation
        if enable_diarization is not None:
            self.enable_diarization = enable_diarization
        if diarization_speaker_count is not None:
            self.diarization_speaker_count = diarization_speaker_count

        logger.info(
            f"Settings updated: punctuation={self.enable_punctuation}, "
            f"diarization={self.enable_diarization}, "
            f"speaker_count={self.diarization_speaker_count}"
        )


def create_engine(
    language_code: str = "ja-JP",
    sample_rate: int = 16000,
    enable_punctuation: bool = True,
    enable_diarization: bool = False,
    diarization_speaker_count: int = 2,
) -> GoogleSTTEngine:
    """
    Create and initialize a Google STT engine.

    Args:
        language_code: Language code for recognition
        sample_rate: Audio sample rate in Hz
        enable_punctuation: Enable automatic punctuation
        enable_diarization: Enable speaker diarization
        diarization_speaker_count: Expected number of speakers

    Returns:
        Initialized GoogleSTTEngine instance
    """
    engine = GoogleSTTEngine(
        language_code=language_code,
        sample_rate=sample_rate,
        enable_punctuation=enable_punctuation,
        enable_diarization=enable_diarization,
        diarization_speaker_count=diarization_speaker_count,
    )
    engine.load()
    return engine
