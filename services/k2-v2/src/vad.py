"""
Voice Activity Detection (VAD) using Silero VAD.

This module provides VAD functionality to segment audio into speech chunks.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms at 16kHz


@dataclass
class VADSegment:
    """A segment of speech detected by VAD."""

    audio: np.ndarray
    start_time: float
    end_time: float
    is_speech: bool


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model.

    This class wraps the Silero VAD model and provides methods for
    detecting speech segments in audio streams.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        sample_rate: int = SAMPLE_RATE,
    ):
        """
        Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold
            min_speech_duration_ms: Minimum speech duration to consider
            min_silence_duration_ms: Minimum silence duration to split segments
            speech_pad_ms: Padding to add around speech segments
            sample_rate: Audio sample rate
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        self.model = None
        self._reset_state()

    def _reset_state(self):
        """Reset internal state."""
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.speech_buffer = []
        self.ring_buffer = deque(maxlen=int(self.speech_pad_ms * self.sample_rate / 1000))

    def load_model(self):
        """Load the Silero VAD model."""
        import torch

        logger.info("Loading Silero VAD model...")

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        self.model = model
        self.get_speech_timestamps = utils[0]
        self._reset_state()

        logger.info("Silero VAD model loaded successfully")

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[VADSegment]:
        """
        Process an audio chunk and detect speech.

        Args:
            audio_chunk: Audio data as numpy array (float32, mono)

        Returns:
            VADSegment if speech is detected, None otherwise
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure correct format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Get speech probability
        audio_tensor = torch.from_numpy(audio_chunk)
        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        chunk_duration = len(audio_chunk) / self.sample_rate
        current_time = self.current_sample / self.sample_rate

        # Update sample counter
        self.current_sample += len(audio_chunk)

        # State machine for speech detection
        if speech_prob >= self.threshold:
            if not self.triggered:
                # Speech started
                self.triggered = True
                self.speech_start = current_time - (self.speech_pad_ms / 1000)
                # Add ring buffer content
                self.speech_buffer = list(self.ring_buffer)

            self.speech_buffer.append(audio_chunk)
            self.temp_end = 0

        else:
            if self.triggered:
                # Potential end of speech
                self.temp_end += len(audio_chunk)
                self.speech_buffer.append(audio_chunk)

                silence_duration_ms = (self.temp_end / self.sample_rate) * 1000

                if silence_duration_ms >= self.min_silence_duration_ms:
                    # Speech ended
                    self.triggered = False
                    speech_end = current_time + chunk_duration

                    # Check minimum duration
                    speech_duration_ms = (speech_end - self.speech_start) * 1000

                    if speech_duration_ms >= self.min_speech_duration_ms:
                        # Combine all buffered audio
                        combined_audio = np.concatenate(self.speech_buffer)

                        segment = VADSegment(
                            audio=combined_audio,
                            start_time=self.speech_start,
                            end_time=speech_end,
                            is_speech=True,
                        )

                        self.speech_buffer = []
                        self.temp_end = 0

                        return segment

                    self.speech_buffer = []
                    self.temp_end = 0

            else:
                # Keep ring buffer for padding
                self.ring_buffer.append(audio_chunk)

        return None

    def reset(self):
        """Reset VAD state for a new audio stream."""
        self._reset_state()
        if self.model is not None:
            self.model.reset_states()


class AudioChunker:
    """
    Chunks audio stream into fixed-size segments with VAD.

    This class combines VAD with chunking to produce speech segments
    suitable for offline transcription.
    """

    def __init__(
        self,
        vad: SileroVAD,
        max_chunk_duration: float = 5.0,
        min_chunk_duration: float = 0.5,
        sample_rate: int = SAMPLE_RATE,
    ):
        """
        Initialize the audio chunker.

        Args:
            vad: SileroVAD instance
            max_chunk_duration: Maximum chunk duration in seconds
            min_chunk_duration: Minimum chunk duration in seconds
            sample_rate: Audio sample rate
        """
        self.vad = vad
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.sample_rate = sample_rate

        self.buffer = np.array([], dtype=np.float32)
        self.buffer_start_time = 0.0

    def add_audio(self, audio: np.ndarray) -> list[VADSegment]:
        """
        Add audio to the buffer and return any complete segments.

        Args:
            audio: Audio data as numpy array

        Returns:
            List of VAD segments ready for transcription
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments = []

        # Process in VAD-sized chunks
        for i in range(0, len(audio), CHUNK_SIZE):
            chunk = audio[i : i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                # Pad last chunk
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

            segment = self.vad.process_chunk(chunk)
            if segment is not None:
                # Split if too long
                if segment.end_time - segment.start_time > self.max_chunk_duration:
                    segments.extend(self._split_segment(segment))
                else:
                    segments.append(segment)

        return segments

    def _split_segment(self, segment: VADSegment) -> list[VADSegment]:
        """Split a long segment into smaller chunks."""
        segments = []
        max_samples = int(self.max_chunk_duration * self.sample_rate)

        audio = segment.audio
        start_time = segment.start_time

        for i in range(0, len(audio), max_samples):
            chunk_audio = audio[i : i + max_samples]
            chunk_duration = len(chunk_audio) / self.sample_rate

            segments.append(
                VADSegment(
                    audio=chunk_audio,
                    start_time=start_time,
                    end_time=start_time + chunk_duration,
                    is_speech=True,
                )
            )
            start_time += chunk_duration

        return segments

    def flush(self) -> Optional[VADSegment]:
        """
        Flush any remaining audio in the VAD buffer.

        Returns:
            Final VAD segment if any speech was buffered
        """
        if self.vad.triggered and self.vad.speech_buffer:
            combined_audio = np.concatenate(self.vad.speech_buffer)
            duration = len(combined_audio) / self.sample_rate

            if duration >= self.min_chunk_duration:
                segment = VADSegment(
                    audio=combined_audio,
                    start_time=self.vad.speech_start,
                    end_time=self.vad.speech_start + duration,
                    is_speech=True,
                )
                self.vad.reset()
                return segment

        self.vad.reset()
        return None

    def reset(self):
        """Reset the chunker state."""
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_start_time = 0.0
        self.vad.reset()
