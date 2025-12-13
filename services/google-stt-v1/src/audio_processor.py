"""
Audio processing utilities for Google Speech-to-Text.

Handles audio format conversion and message parsing.
"""

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AudioConfig:
    """Audio configuration."""

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit


def parse_audio_message(message: bytes) -> Tuple[int, np.ndarray]:
    """
    Parse an audio message from the WebSocket client.

    The message format is:
    - First 4 bytes: sample rate (little-endian uint32)
    - Remaining bytes: 16-bit PCM audio data

    Args:
        message: Raw bytes from WebSocket

    Returns:
        Tuple of (sample_rate, audio_data as float32 numpy array)
    """
    if len(message) < 4:
        raise ValueError("Message too short")

    # Extract sample rate from header
    sample_rate = struct.unpack("<I", message[:4])[0]

    # Extract audio data
    audio_bytes = message[4:]

    # Convert to numpy array (int16)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

    # Convert to float32 normalized to [-1, 1]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return sample_rate, audio_float32


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """
    Convert float32 audio to 16-bit PCM bytes.

    Args:
        audio: Float32 audio data normalized to [-1, 1]

    Returns:
        Raw bytes of 16-bit PCM audio
    """
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16.tobytes()


def resample_audio(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int = 16000,
) -> np.ndarray:
    """
    Simple linear interpolation resampling.

    Args:
        audio: Input audio data
        original_rate: Original sample rate
        target_rate: Target sample rate

    Returns:
        Resampled audio data
    """
    if original_rate == target_rate:
        return audio

    duration = len(audio) / original_rate
    new_length = int(duration * target_rate)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
