"""
Netzach (Victory) - External service integration and API management.
"""

import logging
from typing import Dict, Optional

import httpx

from ..config import settings
from ..utils.circuit_breaker import circuit_breaker
from hermes.utils.logging import logger

logger = logging.getLogger(__name__)


class Netzach:
    """Victory sphere - Manages external integrations."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=settings.timeout_seconds)
        self.adapter = APIAdapter()

    async def integrate(self, data: Dict) -> Dict:
        """Process external integrations for the request."""
        try:
            if not settings.allow_external:
                return {"external": {}, "status": "external_disabled"}

            services = self._determine_services(data)
            results = {}

            for service in services:
                results[service] = await self._call_service(service, data)

            return {"external": results, "status": "success"}

        except Exception as e:
            logger.error(f"Integration failed: {str(e)}")
            return {"external": {}, "status": "integration_failed"}

    def _determine_services(self, data: Dict) -> list:
        """Determine which external services to call based on request data."""
        services = []
        if settings.firecrawl_api_key and "query" in data:
            services.append("firecrawl")
        return services

    @circuit_breaker(lambda service, _: {"error": f"{service}_unavailable"})
    async def _call_service(self, service: str, data: Dict) -> Dict:
        """Call external service with adaptive retry logic."""
        endpoint, request_data = self.adapter.prepare_request(service, data)

        for attempt in range(3):
            try:
                response = await self.client.post(
                    endpoint, headers=request_data["headers"], json=request_data["json"]
                )
                response.raise_for_status()
                return self.adapter.parse_response(service, response.json())
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                logger.warning(f"Retry {attempt+1} for {service}: {str(e)}")

        raise ConnectionError(f"Service {service} unavailable")


class APIAdapter:
    """Adapts requests to various external API formats."""

    SERVICES = {
        "firecrawl": {
            "endpoint": "https://firecrawl.com/api/v1/search",
            "auth_header": "Bearer {api_key}",
            "payload_map": {"query": "query", "context": "metadata"},
        }
    }

    def prepare_request(self, service: str, data: Dict) -> tuple:
        """Prepare service-specific request format."""
        config = self.SERVICES[service]
        endpoint = config["endpoint"]
        headers = {
            "Authorization": config["auth_header"].format(
                api_key=settings.firecrawl_api_key
            )
        }
        payload = {
            config["payload_map"].get(k, k): v
            for k, v in data.items()
            if k in config["payload_map"]
        }
        return endpoint, {"headers": headers, "json": payload}

    def parse_response(self, service: str, response: Dict) -> Dict:
        """Parse service-specific response format."""
        if service == "firecrawl":
            return {
                "content": response.get("result", "No result returned."),
                "metadata": response.get("metadata", {}),
            }
        return response
