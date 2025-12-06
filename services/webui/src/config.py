"""
Configuration for ASR backend models.

Defines the available models and their connection settings.
"""

import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a single ASR model."""

    id: str
    name: str
    description: str
    url: str
    speed: str  # "fast" or "standard"


# Model configurations
# URLs use Docker service names for internal networking
MODELS: dict[str, ModelConfig] = {
    "k2-v2": ModelConfig(
        id="k2-v2",
        name="K2-v2",
        description="Sherpa-ONNX Transducer (159M params, オフライン最速)",
        url=os.environ.get("K2_V2_URL", "http://k2-v2:8000"),
        speed="fast",
    ),
    "espnet-v2": ModelConfig(
        id="espnet-v2",
        name="ESPnet-v2",
        description="ESPnet Conformer-Transducer (119M params)",
        url=os.environ.get("ESPNET_V2_URL", "http://espnet-v2:8000"),
        speed="standard",
    ),
    "espnet-v2-onnx": ModelConfig(
        id="espnet-v2-onnx",
        name="ESPnet-v2 ONNX",
        description="ESPnet + ONNX Runtime (119M params, 高速)",
        url=os.environ.get("ESPNET_V2_ONNX_URL", "http://espnet-v2-onnx:8000"),
        speed="fast",
    ),
    "google-stt": ModelConfig(
        id="google-stt",
        name="Google Speech-to-Text",
        description="Google Cloud Speech-to-Text API (Streaming), 🚀全モデル最速・⏫️最高品質",
        url=os.environ.get("GOOGLE_STT_URL", "http://google-stt:8000"),
        speed="fast",
    ),
}

# Default model to use
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "k2-v2")


def get_model(model_id: str) -> ModelConfig | None:
    """Get model configuration by ID."""
    return MODELS.get(model_id)


def get_all_models() -> list[ModelConfig]:
    """Get all available model configurations."""
    return list(MODELS.values())
