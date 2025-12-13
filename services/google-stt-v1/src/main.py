"""
Main entry point for Google Speech-to-Text ASR service.
"""

import argparse
import logging
import os

import uvicorn

from .server import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Google Speech-to-Text Real-time ASR Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=os.environ.get("LANGUAGE_CODE", "ja-JP"),
        help="Language code for recognition (default: ja-JP)",
    )
    parser.add_argument(
        "--enable-punctuation",
        action="store_true",
        default=os.environ.get("ENABLE_PUNCTUATION", "true").lower() == "true",
        help="Enable automatic punctuation (default: true)",
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="Disable automatic punctuation",
    )
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        default=os.environ.get("ENABLE_DIARIZATION", "false").lower() == "true",
        help="Enable speaker diarization (default: false)",
    )
    parser.add_argument(
        "--speaker-count",
        type=int,
        default=int(os.environ.get("DIARIZATION_SPEAKER_COUNT", "2")),
        help="Expected number of speakers for diarization (default: 2)",
    )

    args = parser.parse_args()

    # Handle punctuation flag
    enable_punctuation = args.enable_punctuation and not args.no_punctuation

    # Check for credentials
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        logger.info(f"Using credentials from: {creds_path}")
    else:
        logger.warning(
            "GOOGLE_APPLICATION_CREDENTIALS not set. "
            "Make sure credentials are configured."
        )

    # Create and run the app
    app = create_app(
        language_code=args.language,
        sample_rate=16000,
        enable_punctuation=enable_punctuation,
        enable_diarization=args.enable_diarization,
        diarization_speaker_count=args.speaker_count,
    )

    logger.info(f"Starting Google STT server on {args.host}:{args.port}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Punctuation: {enable_punctuation}")
    logger.info(f"Diarization: {args.enable_diarization} (speakers: {args.speaker_count})")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
