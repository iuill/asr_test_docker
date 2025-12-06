"""
Transcription Engine using espnet_onnx for reazonspeech-espnet-v2.

This module provides a wrapper around espnet_onnx for offline speech recognition
using the reazonspeech-espnet-v2 model exported to ONNX format.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
MODEL_TAG = "reazon-research/reazonspeech-espnet-v2"
SAMPLE_RATE = 16000


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""

    text: str
    duration: float
    is_final: bool = True


class TranscriptionEngine:
    """
    Speech recognition engine using espnet_onnx and reazonspeech-espnet-v2.

    This engine uses an ONNX-exported offline (non-streaming) model,
    so audio is processed in chunks after VAD segmentation.
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
            model_dir: Directory containing the ONNX model files. If None, uses default cache.
            num_threads: Number of threads for inference
        """
        if self._initialized:
            return

        self.device = device
        self.num_threads = num_threads
        self.model_dir = model_dir
        self.speech2text = None

        self._initialized = True

    def _get_onnx_cache_dir(self) -> str:
        """Get the default ONNX model cache directory."""
        cache_dir = os.environ.get("ESPNET_ONNX_CACHE", Path.home() / ".cache" / "espnet_onnx")
        return str(cache_dir)

    def _export_model_if_needed(self) -> str:
        """
        Export the ESPnet model to ONNX format if not already exported.

        Returns:
            Path to the exported model directory (relative to cache_dir)
        """
        from espnet_onnx.export import ASRModelExport

        # espnet_onnx preserves "/" as subdirectories (e.g., reazon-research/reazonspeech-espnet-v2)
        cache_dir = Path(self._get_onnx_cache_dir())
        model_path = cache_dir / MODEL_TAG  # Uses "/" to create subdirectories

        if model_path.exists() and (model_path / "config.yaml").exists():
            logger.info(f"Using cached ONNX model from {model_path}")
            return MODEL_TAG

        logger.info("Exporting reazonspeech-espnet-v2 to ONNX format...")
        logger.info("This may take a few minutes on first run...")

        try:
            exporter = ASRModelExport(cache_dir=str(cache_dir))
            # Export from pretrained model with quantization for better performance
            # tag_name is the espnet_model_zoo tag for the pretrained model
            exporter.export_from_pretrained(
                tag_name=MODEL_TAG,
                quantize=True,
            )
            logger.info(f"Model exported successfully to {model_path}")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

        return MODEL_TAG

    def load_model(self) -> None:
        """Load the speech recognition model."""
        logger.info("Loading reazonspeech-espnet-v2 ONNX model...")

        try:
            from espnet_onnx import Speech2Text

            # Export model if needed (first run)
            tag_name = self._export_model_if_needed()
            cache_dir = self._get_onnx_cache_dir()

            # Set up providers based on device
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            # Load the ONNX model
            # Use model_dir to specify the exact path to the exported model
            model_dir = str(Path(cache_dir) / tag_name)
            self.speech2text = Speech2Text(
                model_dir=model_dir,
                providers=providers,
            )

            logger.info("ONNX model loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import espnet_onnx: {e}")
            logger.error("Please ensure espnet-onnx is installed")
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
        if self.speech2text is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed (espnet_onnx expects values in [-1, 1])
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, SAMPLE_RATE)

        try:
            # Perform transcription
            # espnet_onnx Speech2Text returns a list of (text, token, token_int, hyp) tuples
            nbest = self.speech2text(audio)

            if nbest and len(nbest) > 0:
                # Get the best hypothesis
                text = nbest[0][0] if nbest[0] else ""
            else:
                text = ""

            duration = len(audio) / SAMPLE_RATE

            return TranscriptionResult(
                text=text.strip() if text else "",
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
        return self.speech2text is not None


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
