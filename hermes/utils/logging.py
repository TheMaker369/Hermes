"""
Logging configuration for the Hermes AI System.
"""

import logging
import sys
from typing import Optional
from hermes.utils.logging import logger


def setup_logging(level: Optional[int] = None):
    """
    Configure logging for the Hermes system.

    Args:
        level: Optional logging level (defaults to INFO if not specified)
    """
    if level is None:
        level = logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure stream handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Specific logger for Hermes
    hermes_logger = logging.getLogger("hermes")
    hermes_logger.setLevel(level)

    # Suppress excessive logging from dependencies
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ray").setLevel(logging.WARNING)


from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add custom stderr handler
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Optionally, add file logging
logger.add(
    "logs/file_{time}.log", rotation="500 MB", retention="10 days", compression="zip"
)

try:
    result = external_api_call(data)
except Exception as e:
    logger.error(f"Error during external_api_call with data {data}: {e}")
    result = None  # or use a default value
