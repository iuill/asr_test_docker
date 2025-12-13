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
    append_newline_on_final: bool = False  # 確定テキストの末尾に改行を追加するか


# Model configurations
# URLs use Docker service names for internal networking
MODELS: dict[str, ModelConfig] = {
    "k2-v2": ModelConfig(
        id="k2-v2",
        name="reazonspeech-k2-v2",
        description="Sherpa-ONNX Transducer (159M params, ローカル実行で高速)",
        url=os.environ.get("K2_V2_URL", "http://k2-v2:8000"),
        speed="fast",
        append_newline_on_final=True,
    ),
    "espnet-v2": ModelConfig(
        id="espnet-v2",
        name="reazonspeech-espnet-v2",
        description="ESPnet Conformer-Transducer (119M params)",
        url=os.environ.get("ESPNET_V2_URL", "http://espnet-v2:8000"),
        speed="standard",
        append_newline_on_final=True,
    ),
    "espnet-v2-onnx": ModelConfig(
        id="espnet-v2-onnx",
        name="reazonspeech-espnet-v2 (ONNX)",
        description="ESPnet + ONNX Runtime (119M params)",
        url=os.environ.get("ESPNET_V2_ONNX_URL", "http://espnet-v2-onnx:8000"),
        speed="fast",
        append_newline_on_final=True,
    ),
    "google-stt": ModelConfig(
        id="google-stt",
        name="Google STT (default)",
        description="Google Cloud Speech-to-Text API (default model)",
        url=os.environ.get("GOOGLE_STT_URL", "http://google-stt:8000"),
        speed="fast",
        append_newline_on_final=True,
    ),
    "google-stt-chirp2": ModelConfig(
        id="google-stt-chirp2",
        name="Google STT V2 (Chirp 2)",
        description="Google Cloud Speech-to-Text V2 API (Chirp 2, asia-southeast1)",
        url=os.environ.get("GOOGLE_STT_CHIRP2_URL", "http://google-stt-chirp2:8000"),
        speed="fast",
        append_newline_on_final=True,
    ),
    "google-stt-chirp3": ModelConfig(
        id="google-stt-chirp3",
        name="Google STT V2 (Chirp 3)",
        description="Google Cloud Speech-to-Text V2 API (Chirp 3, asia-south1), ⏫️高品質",
        url=os.environ.get("GOOGLE_STT_CHIRP3_URL", "http://google-stt-chirp3:8000"),
        speed="fast",
        append_newline_on_final=True,
    ),
    "openai-stt": ModelConfig(
        id="openai-stt",
        name="OpenAI gpt-4o-transcribe",
        description="OpenAI Realtime API (gpt-4o-transcribe), ⏫️高品質",
        url=os.environ.get("OPENAI_STT_URL", "http://openai-stt:8000"),
        speed="fast",
        append_newline_on_final=True,
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
