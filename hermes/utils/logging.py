"""
Logging configuration for the Hermes AI System.
"""

import logging
import sys
from typing import Optional

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
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
