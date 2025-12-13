"""
Azure Speech-to-Text Transcription Engine.

Provides streaming speech recognition using Azure AI Speech SDK
with PushAudioInputStream for real-time transcription.
"""

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

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


class AzureSTTEngine:
    """
    Azure Speech-to-Text streaming transcription engine.

    This engine uses Azure AI Speech SDK with PushAudioInputStream
    for real-time speech recognition.
    """

    def __init__(
        self,
        speech_key: str,
        speech_region: str,
        language_code: str = "ja-JP",
        sample_rate: int = 16000,
        enable_punctuation: bool = True,
    ):
        """
        Initialize the Azure STT engine.

        Args:
            speech_key: Azure Speech resource key
            speech_region: Azure Speech resource region (e.g., "japaneast")
            language_code: Language code for recognition (default: ja-JP)
            sample_rate: Audio sample rate in Hz (default: 16000)
            enable_punctuation: Enable automatic punctuation
        """
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.enable_punctuation = enable_punctuation

        self._speech_config: Optional[speechsdk.SpeechConfig] = None
        self._loaded = False

        # Streaming state
        self._push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None
        self._audio_config: Optional[speechsdk.audio.AudioConfig] = None
        self._recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self._recognition_thread: Optional[threading.Thread] = None
        self._result_queue: Optional[asyncio.Queue] = None
        self._is_streaming = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: threading.Event = threading.Event()
        self._result_index: int = 0

    def load(self) -> None:
        """Load the Azure Speech SDK client."""
        if self._loaded:
            return

        try:
            if not self.speech_key or not self.speech_region:
                raise ValueError(
                    "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set"
                )

            # Create speech config
            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region,
            )

            # Set language
            self._speech_config.speech_recognition_language = self.language_code

            # Configure output format for detailed results
            self._speech_config.output_format = speechsdk.OutputFormat.Detailed

            self._loaded = True
            logger.info(
                f"Azure Speech SDK initialized "
                f"(region: {self.speech_region}, language: {self.language_code})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Azure Speech SDK: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if the engine is loaded."""
        return self._loaded

    def _create_recognizer(self) -> speechsdk.SpeechRecognizer:
        """Create a new speech recognizer with push stream."""
        # Create audio format (16-bit PCM, mono)
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate,
            bits_per_sample=16,
            channels=1,
        )

        # Create push stream
        self._push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=audio_format
        )

        # Create audio config from push stream
        self._audio_config = speechsdk.audio.AudioConfig(
            stream=self._push_stream
        )

        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=self._audio_config,
        )

        return recognizer

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle intermediate recognition results."""
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = evt.result.text
            if text.strip():
                # Get timing info (in ticks, 100ns units)
                offset_ticks = evt.result.offset
                duration_ticks = evt.result.duration
                start_time = offset_ticks / 10_000_000  # Convert to seconds
                end_time = start_time + (duration_ticks / 10_000_000)

                result = TranscriptionResult(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    is_partial=True,
                    stability=0.5,  # Intermediate stability
                    confidence=0.0,  # Not available for intermediate
                    result_index=self._result_index,
                )

                if self._loop and self._result_queue:
                    asyncio.run_coroutine_threadsafe(
                        self._result_queue.put(result),
                        self._loop,
                    )

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle final recognition results."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            if text.strip():
                # Get timing info
                offset_ticks = evt.result.offset
                duration_ticks = evt.result.duration
                start_time = offset_ticks / 10_000_000
                end_time = start_time + (duration_ticks / 10_000_000)

                # Get confidence from detailed results if available
                confidence = 0.0
                try:
                    json_result = evt.result.properties.get(
                        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                    )
                    if json_result:
                        import json
                        result_data = json.loads(json_result)
                        nbest = result_data.get("NBest", [])
                        if nbest:
                            confidence = nbest[0].get("Confidence", 0.0)
                except Exception:
                    pass

                result = TranscriptionResult(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    is_partial=False,
                    stability=1.0,  # Final result
                    confidence=confidence,
                    result_index=self._result_index,
                )
                self._result_index += 1

                if self._loop and self._result_queue:
                    asyncio.run_coroutine_threadsafe(
                        self._result_queue.put(result),
                        self._loop,
                    )

        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug(f"No speech recognized: {evt.result.no_match_details}")

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        """Handle recognition cancellation."""
        if evt.reason == speechsdk.CancellationReason.Error:
            error_msg = f"Recognition error: {evt.error_details}"
            logger.error(error_msg)

            if self._loop and self._result_queue:
                error_result = TranscriptionResult(
                    text=f"[Error: {evt.error_details}]",
                    start_time=0.0,
                    end_time=0.0,
                    is_partial=False,
                )
                asyncio.run_coroutine_threadsafe(
                    self._result_queue.put(error_result),
                    self._loop,
                )
        elif evt.reason == speechsdk.CancellationReason.EndOfStream:
            logger.info("End of audio stream")

    def _on_session_stopped(self, evt: speechsdk.SessionEventArgs) -> None:
        """Handle session stopped event."""
        logger.debug("Recognition session stopped")
        self._stop_event.set()

    def _recognition_thread_func(self) -> None:
        """Run continuous recognition in a separate thread."""
        try:
            # Create recognizer
            self._recognizer = self._create_recognizer()

            # Connect event handlers
            self._recognizer.recognizing.connect(self._on_recognizing)
            self._recognizer.recognized.connect(self._on_recognized)
            self._recognizer.canceled.connect(self._on_canceled)
            self._recognizer.session_stopped.connect(self._on_session_stopped)

            # Start continuous recognition
            self._recognizer.start_continuous_recognition()
            logger.info("Started continuous recognition")

            # Wait until stopped
            self._stop_event.wait()

        except Exception as e:
            logger.error(f"Recognition thread error: {e}")
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
        self._is_streaming = True
        self._stop_event.clear()
        self._result_index = 0

        # Start the recognition thread
        self._recognition_thread = threading.Thread(
            target=self._recognition_thread_func,
            daemon=True,
        )
        self._recognition_thread.start()

        return self._result_queue

    def add_audio(self, audio_bytes: bytes) -> None:
        """
        Add audio data to the streaming queue.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM)
        """
        if self._is_streaming and self._push_stream:
            self._push_stream.write(audio_bytes)

    def stop_streaming(self) -> None:
        """Stop streaming recognition."""
        if not self._is_streaming:
            return

        self._is_streaming = False

        # Close the push stream to signal end of audio
        if self._push_stream:
            self._push_stream.close()

        # Stop continuous recognition
        if self._recognizer:
            try:
                self._recognizer.stop_continuous_recognition()
            except Exception as e:
                logger.warning(f"Error stopping recognition: {e}")

        # Signal the thread to stop
        self._stop_event.set()

        # Wait for thread to finish
        if self._recognition_thread and self._recognition_thread.is_alive():
            self._recognition_thread.join(timeout=5.0)

        self._recognition_thread = None
        self._recognizer = None
        self._push_stream = None
        self._audio_config = None
        self._result_queue = None
        self._loop = None

    def reset(self) -> None:
        """Reset the engine state."""
        self.stop_streaming()
        self._result_index = 0

    def update_settings(
        self,
        enable_punctuation: Optional[bool] = None,
    ) -> None:
        """
        Update engine settings.

        Args:
            enable_punctuation: Enable automatic punctuation
        """
        if enable_punctuation is not None:
            self.enable_punctuation = enable_punctuation

        logger.info(f"Settings updated: punctuation={self.enable_punctuation}")


def create_engine(
    speech_key: str,
    speech_region: str,
    language_code: str = "ja-JP",
    sample_rate: int = 16000,
    enable_punctuation: bool = True,
) -> AzureSTTEngine:
    """
    Create and initialize an Azure STT engine.

    Args:
        speech_key: Azure Speech resource key
        speech_region: Azure Speech resource region
        language_code: Language code for recognition
        sample_rate: Audio sample rate in Hz
        enable_punctuation: Enable automatic punctuation

    Returns:
        Initialized AzureSTTEngine instance
    """
    engine = AzureSTTEngine(
        speech_key=speech_key,
        speech_region=speech_region,
        language_code=language_code,
        sample_rate=sample_rate,
        enable_punctuation=enable_punctuation,
    )
    engine.load()
    return engine
