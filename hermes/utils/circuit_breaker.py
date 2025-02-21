"""
Circuit breaker pattern implementation for external service calls.
"""

import functools
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

def circuit_breaker(fallback: Callable[..., Any] = None):
    """
    Circuit breaker decorator for external service calls.
    
    Args:
        fallback: Function to call when the main function fails
        
    Returns:
        Decorator function that implements circuit breaking
    """
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Circuit breaker activated for {func.__name__}: {str(e)}")
                if fallback:
                    if isinstance(fallback, str):
                        return fallback
                    return fallback(*args)
                return f"Service {func.__name__} unavailable"
        return wrapper
    return decorator
