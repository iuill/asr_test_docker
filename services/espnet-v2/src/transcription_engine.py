"""
Transcription Engine using reazonspeech-espnet-v2.

This module provides a wrapper around the reazonspeech ESPnet model
for offline speech recognition.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "reazon-research/reazonspeech-espnet-v2"
SAMPLE_RATE = 16000


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""

    text: str
    duration: float
    is_final: bool = True


class TranscriptionEngine:
    """
    Speech recognition engine using reazonspeech-espnet-v2.

    This engine uses an offline (non-streaming) model, so audio is processed
    in chunks after VAD segmentation.
    """

    _instance: Optional["TranscriptionEngine"] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to reuse the same model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        device: str = "cpu",
        model_dir: Optional[str] = None,
        num_threads: int = 4,
    ):
        """
        Initialize the transcription engine.

        Args:
            device: Device to use ('cpu' or 'cuda')
            model_dir: Directory containing the model files (not used for ESPnet, model auto-downloaded)
            num_threads: Number of threads for inference (not directly used by ESPnet)
        """
        if self._initialized:
            return

        self.device = device
        self.num_threads = num_threads
        self.model_dir = model_dir
        self.model = None
        self._audio_interface = None

        self._initialized = True

    def load_model(self) -> None:
        """Load the speech recognition model."""
        logger.info("Loading reazonspeech-espnet-v2 model...")

        try:
            # Import reazonspeech ESPnet modules
            from reazonspeech.espnet.asr import load_model

            # Load the model
            # The model is automatically downloaded from HuggingFace
            self.model = load_model(device=self.device)

            logger.info("Model loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import reazonspeech.espnet.asr: {e}")
            logger.error("Please ensure reazonspeech[espnet] is installed")
            raise

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> TranscriptionResult:
        """
        Transcribe audio data.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio

        Returns:
            TranscriptionResult with the transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed (ESPnet expects values in [-1, 1])
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, SAMPLE_RATE)

        try:
            # Import transcribe function and audio_from_numpy
            from reazonspeech.espnet.asr import transcribe, audio_from_numpy

            # Create AudioData object using audio_from_numpy
            audio_data = audio_from_numpy(audio, SAMPLE_RATE)

            # Perform transcription
            result = transcribe(self.model, audio_data)

            duration = len(audio) / SAMPLE_RATE

            return TranscriptionResult(
                text=result.text.strip() if result.text else "",
                duration=duration,
                is_final=True,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            # Return empty result on error
            return TranscriptionResult(
                text="",
                duration=len(audio) / SAMPLE_RATE,
                is_final=True,
            )

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Simple resampling using linear interpolation.

        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None


def create_engine(
    device: str = "cpu",
    model_dir: Optional[str] = None,
    num_threads: int = 4,
) -> TranscriptionEngine:
    """
    Create and initialize a transcription engine.

    Args:
        device: Device to use ('cpu' or 'cuda')
        model_dir: Directory containing the model files
        num_threads: Number of threads for inference

    Returns:
        Initialized TranscriptionEngine
    """
    engine = TranscriptionEngine(
        device=device,
        model_dir=model_dir,
        num_threads=num_threads,
    )
    engine.load_model()
    return engine
