"""
Main entry point for OpenAI Speech-to-Text ASR service.
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
        description="OpenAI Speech-to-Text Real-time ASR Server (gpt-4o-transcribe)"
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
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-transcribe"),
        choices=["gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
        help="Model to use (default: gpt-4o-transcribe)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=os.environ.get("LANGUAGE_CODE", "ja"),
        help="Language code for recognition (default: ja)",
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key found")
    else:
        logger.error(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it to your OpenAI API key."
        )
        return

    # Create and run the app
    app = create_app(
        model=args.model,
        language=args.language,
        sample_rate=24000,
    )

    logger.info(f"Starting OpenAI STT server on {args.host}:{args.port}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Language: {args.language}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
