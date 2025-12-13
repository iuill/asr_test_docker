"""
Google Speech-to-Text V2 API Transcription Engine.

Provides streaming speech recognition using Google Cloud Speech-to-Text V2 API
with support for Chirp 2/3 models.
"""

import asyncio
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Generator, Optional

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

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


class GoogleSTTV2Engine:
    """
    Google Speech-to-Text V2 streaming transcription engine.

    This engine uses Google Cloud Speech-to-Text V2 API for real-time
    speech recognition with streaming support for Chirp 2/3 models.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us",
        language_code: str = "ja-JP",
        sample_rate: int = 16000,
        enable_punctuation: bool = True,
        enable_diarization: bool = False,
        diarization_speaker_count: int = 2,
        model: str = "chirp_2",
    ):
        """
        Initialize the Google STT V2 engine.

        Args:
            project_id: Google Cloud project ID
            location: API location (e.g., "us", "eu", "us-central1")
            language_code: Language code for recognition (default: ja-JP)
            sample_rate: Audio sample rate in Hz
            enable_punctuation: Enable automatic punctuation
            enable_diarization: Enable speaker diarization
            diarization_speaker_count: Expected number of speakers (for diarization)
            model: Recognition model to use ("chirp_2", "chirp_3")
        """
        self.project_id = project_id
        self.location = location
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.enable_punctuation = enable_punctuation
        self.enable_diarization = enable_diarization
        self.diarization_speaker_count = diarization_speaker_count
        self.model = model
        self._client: Optional[SpeechClient] = None
        self._recognizer: Optional[str] = None
        self._loaded = False

        # Streaming state
        self._audio_queue: queue.Queue = queue.Queue()
        self._streaming_thread: Optional[threading.Thread] = None
        self._result_queue: asyncio.Queue = None
        self._is_streaming = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def load(self) -> None:
        """Load the Google Speech V2 client."""
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

            # Create client with regional endpoint
            api_endpoint = f"{self.location}-speech.googleapis.com"
            self._client = SpeechClient(
                client_options=ClientOptions(api_endpoint=api_endpoint)
            )

            # Create recognizer path
            self._recognizer = self._client.recognizer_path(
                self.project_id, self.location, "_"
            )

            self._loaded = True
            logger.info(
                f"Google Speech-to-Text V2 client initialized "
                f"(endpoint: {api_endpoint}, model: {self.model})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Google STT V2 client: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if the engine is loaded."""
        return self._loaded

    def get_recognition_config(self) -> cloud_speech.RecognitionConfig:
        """Get the recognition configuration."""
        # Build recognition features
        features = cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=self.enable_punctuation,
        )

        # Add diarization config if enabled
        if self.enable_diarization:
            features.diarization_config = cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=1,
                max_speaker_count=self.diarization_speaker_count,
            )

        # Build recognition config
        # Use explicit decoding config for LINEAR16 audio
        explicit_config = cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=1,
        )

        # Use single language code to disable automatic language detection
        # Multiple language_codes would enable auto-detection
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=explicit_config,
            language_codes=[self.language_code],  # Single language = no auto-detection
            model=self.model,
            features=features,
        )

        logger.info(f"Using recognition model: {self.model}, language: {self.language_code} (auto-detection: disabled)")
        return recognition_config

    def get_streaming_config(self) -> cloud_speech.StreamingRecognitionConfig:
        """Get the streaming recognition configuration."""
        recognition_config = self.get_recognition_config()

        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )

        return streaming_config

    def _create_request_generator(self) -> Generator[cloud_speech.StreamingRecognizeRequest, None, None]:
        """Generate streaming requests."""
        # First request: configuration
        config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self._recognizer,
            streaming_config=self.get_streaming_config(),
        )
        yield config_request

        # Subsequent requests: audio data
        while self._is_streaming:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None:
                    # End signal
                    break
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
            except queue.Empty:
                continue

    def _streaming_recognize_thread(self) -> None:
        """Run streaming recognition in a separate thread."""
        try:
            # Create the streaming request generator
            requests = self._create_request_generator()

            # Start streaming recognition
            responses = self._client.streaming_recognize(requests=requests)

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

                    # Extract confidence
                    confidence = getattr(alternative, 'confidence', 0.0)

                    # Extract result_end_time
                    result_end_time = getattr(result, 'result_end_offset', None)
                    if result_end_time:
                        end_time = result_end_time.seconds + result_end_time.microseconds / 1_000_000
                    else:
                        end_time = 0.0

                    # Check if this is a final result
                    is_final = getattr(result, 'is_final', False)

                    # Extract speaker tag if diarization is enabled
                    # V2 API uses different structure for word info
                    if self.enable_diarization and is_final:
                        words = getattr(alternative, 'words', [])
                        if words:
                            last_word = words[-1]
                            speaker_tag = getattr(last_word, 'speaker_label', '') or 0
                            # Convert speaker label to int if it's a string like "1"
                            if isinstance(speaker_tag, str) and speaker_tag.isdigit():
                                speaker_tag = int(speaker_tag)

                    if text.strip():
                        # V2 doesn't have stability field, use 0.0 for interim, 1.0 for final
                        stability = 1.0 if is_final else 0.0

                        transcription_result = TranscriptionResult(
                            text=text,
                            start_time=0.0,
                            end_time=end_time,
                            is_partial=not is_final,
                            speaker_tag=speaker_tag if isinstance(speaker_tag, int) else 0,
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
    project_id: str,
    location: str = "us",
    language_code: str = "ja-JP",
    sample_rate: int = 16000,
    enable_punctuation: bool = True,
    enable_diarization: bool = False,
    diarization_speaker_count: int = 2,
    model: str = "chirp_2",
) -> GoogleSTTV2Engine:
    """
    Create and initialize a Google STT V2 engine.

    Args:
        project_id: Google Cloud project ID
        location: API location (e.g., "us", "eu", "us-central1")
        language_code: Language code for recognition
        sample_rate: Audio sample rate in Hz
        enable_punctuation: Enable automatic punctuation
        enable_diarization: Enable speaker diarization
        diarization_speaker_count: Expected number of speakers
        model: Recognition model to use ("chirp_2", "chirp_3")

    Returns:
        Initialized GoogleSTTV2Engine instance
    """
    engine = GoogleSTTV2Engine(
        project_id=project_id,
        location=location,
        language_code=language_code,
        sample_rate=sample_rate,
        enable_punctuation=enable_punctuation,
        enable_diarization=enable_diarization,
        diarization_speaker_count=diarization_speaker_count,
        model=model,
    )
    engine.load()
    return engine
