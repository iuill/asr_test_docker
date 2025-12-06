"""
Transcription Engine using sherpa-onnx and reazonspeech-k2-v2.

This module provides a wrapper around sherpa-onnx for offline speech recognition
using the reazonspeech-k2-v2 model.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
MODEL_REPO = "reazon-research/reazonspeech-k2-v2"
SAMPLE_RATE = 16000


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""

    text: str
    duration: float
    is_final: bool = True


class TranscriptionEngine:
    """
    Speech recognition engine using sherpa-onnx and reazonspeech-k2-v2.

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
            device: Device to use ('cpu', 'cuda', or 'coreml')
            model_dir: Directory containing the model files. If None, downloads from HuggingFace.
            num_threads: Number of threads for inference
        """
        if self._initialized:
            return

        self.device = device
        self.num_threads = num_threads
        self.model_dir = model_dir or self._get_default_model_dir()
        self.recognizer = None

        self._initialized = True

    def _get_default_model_dir(self) -> str:
        """Get the default model directory path."""
        cache_dir = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        return str(Path(cache_dir) / "reazonspeech-k2-v2")

    def _download_model(self) -> dict:
        """
        Download model files from HuggingFace Hub.

        Returns:
            Dictionary with paths to model files
        """
        from huggingface_hub import hf_hub_download

        model_dir = Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Actual file names in the HuggingFace repository
        files = {
            "encoder": "encoder-epoch-99-avg-1.onnx",
            "decoder": "decoder-epoch-99-avg-1.onnx",
            "joiner": "joiner-epoch-99-avg-1.onnx",
            "tokens": "tokens.txt",
        }

        paths = {}
        for key, filename in files.items():
            local_path = model_dir / filename
            if not local_path.exists():
                logger.info(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    local_dir=str(model_dir),
                )
            paths[key] = str(local_path)

        return paths

    def load_model(self) -> None:
        """Load the speech recognition model."""
        import sherpa_onnx

        logger.info("Loading reazonspeech-k2-v2 model...")

        # Download model if needed
        model_paths = self._download_model()

        # Configure the recognizer
        config = sherpa_onnx.OfflineRecognizerConfig(
            feat_config=sherpa_onnx.OfflineFeatureExtractorConfig(
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
            ),
            model_config=sherpa_onnx.OfflineModelConfig(
                transducer=sherpa_onnx.OfflineTransducerModelConfig(
                    encoder=model_paths["encoder"],
                    decoder=model_paths["decoder"],
                    joiner=model_paths["joiner"],
                ),
                tokens=model_paths["tokens"],
                num_threads=self.num_threads,
                provider="cuda" if self.device == "cuda" else "cpu",
            ),
        )

        self.recognizer = sherpa_onnx.OfflineRecognizer(config)
        logger.info("Model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> TranscriptionResult:
        """
        Transcribe audio data.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio

        Returns:
            TranscriptionResult with the transcribed text
        """
        if self.recognizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, SAMPLE_RATE)

        # Create stream and process
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self.recognizer.decode_stream(stream)

        result = stream.result
        duration = len(audio) / SAMPLE_RATE

        return TranscriptionResult(
            text=result.text.strip(),
            duration=duration,
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
        return self.recognizer is not None


def create_engine(
    device: str = "cpu",
    model_dir: Optional[str] = None,
    num_threads: int = 4,
) -> TranscriptionEngine:
    """
    Create and initialize a transcription engine.

    Args:
        device: Device to use ('cpu', 'cuda', or 'coreml')
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
