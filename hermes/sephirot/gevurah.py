"""
Gevurah (Severity) - Security, validation, and constraints.
"""

import logging
import re
from typing import Dict, Optional

from fastapi import HTTPException

from ..config import settings
from ..utils.circuit_breaker import circuit_breaker
from hermes.utils.logging import logger

logger = logging.getLogger(__name__)


class Gevurah:
    """Severity sphere - Implements security and validation."""

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.sanitizer = Sanitizer()

    async def validate(self, request: Dict) -> Dict:
        """Validate and sanitize incoming requests."""
        try:
            # Input validation
            self._validate_structure(request)

            # Security checks
            await self.sanitizer.sanitize_input(request)

            # Rate limiting
            if not self.rate_limiter.check(request):
                raise HTTPException(429, "Too many requests")

            return request

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise HTTPException(400, "Invalid request format")

    def _validate_structure(self, request: Dict):
        """Validate request structure against schema."""
        required_keys = {"query", "context", "metadata"}
        if not required_keys.issubset(request.keys()):
            raise ValueError("Missing required request fields")


class Sanitizer:
    """Input sanitization and injection prevention."""

    def __init__(self):
        self.patterns = {
            "sql_injection": re.compile(
                r"(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE){0,1}|INSERT( +INTO){0,1}|MERGE|SELECT|UPDATE|UNION( +ALL){0,1})\b)|(--)|(\\\*)|(\\b(\\d+)(\\s*)(=)(\\s*)(\\d+\\b))",
                re.IGNORECASE,
            ),
            "xss": re.compile(r"<script.*?>.*?</script>", re.IGNORECASE),
        }

    async def sanitize_input(self, data: Dict):
        """Sanitize all input fields recursively."""
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = await self._sanitize_string(value)
            elif isinstance(value, dict):
                await self.sanitize_input(value)

    async def _sanitize_string(self, value: str) -> str:
        """Sanitize individual string values."""
        value = value.replace("\0", "")
        for pattern in self.patterns.values():
            if pattern.search(value):
                raise ValueError("Potentially dangerous input detected")
        return value


class RateLimiter:
    """Adaptive rate limiting based on request patterns."""

    def __init__(self):
        self.counts = {}

    def check(self, request: Dict) -> bool:
        """Check if request is within rate limits."""
        client_id = request.get("metadata", {}).get("client_id", "global")
        # Implement token bucket algorithm
        return True  # Temporary implementation
