"""
Main entry point for Azure Speech-to-Text ASR service with Diarization.
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
        description="Azure Speech-to-Text Real-time ASR Server with Diarization"
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
        "--speech-key",
        type=str,
        default=os.environ.get("AZURE_SPEECH_KEY", ""),
        help="Azure Speech resource key",
    )
    parser.add_argument(
        "--speech-region",
        type=str,
        default=os.environ.get("AZURE_SPEECH_REGION", "japaneast"),
        help="Azure Speech resource region (default: japaneast)",
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

    args = parser.parse_args()

    # Handle punctuation flag
    enable_punctuation = args.enable_punctuation and not args.no_punctuation

    # Validate required parameters
    speech_key = args.speech_key
    speech_region = args.speech_region

    if not speech_key:
        logger.error(
            "Azure Speech key not specified. Set AZURE_SPEECH_KEY environment variable "
            "or use --speech-key argument."
        )
        return

    if not speech_region:
        logger.error(
            "Azure Speech region not specified. Set AZURE_SPEECH_REGION environment variable "
            "or use --speech-region argument."
        )
        return

    # Create and run the app
    app = create_app(
        speech_key=speech_key,
        speech_region=speech_region,
        language_code=args.language,
        sample_rate=16000,
        enable_punctuation=enable_punctuation,
    )

    logger.info(f"Starting Azure STT Diarization server on {args.host}:{args.port}")
    logger.info(f"Speech Region: {speech_region}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Punctuation: {enable_punctuation}")
    logger.info("Speaker Diarization: enabled")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
