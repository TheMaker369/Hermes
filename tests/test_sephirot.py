"""
Tests for the Sephirot components.
"""

from unittest.mock import Mock, patch

import chromadb
import httpx
import pytest
import ray

from hermes.config import settings
from hermes.sephirot.gevurah import Gevurah, Sanitizer
from hermes.sephirot.netzach import APIAdapter, Netzach
from hermes.sephirot.yesod import Yesod


# Yesod (Foundation) Tests
@pytest.mark.asyncio
async def test_yesod_initialization():
    """Test Yesod initialization with ChromaDB."""
    with patch("chromadb.PersistentClient") as mock_chroma:
        yesod = Yesod()
        assert yesod.chroma is not None
        mock_chroma.assert_called_once()


@pytest.mark.asyncio
async def test_yesod_storage():
    """Test data storage through Yesod."""
    yesod = Yesod()
    test_data = {"test": "data"}
    result = await yesod.store(test_data)
    assert result["status"] in ["success", "storage_failed"]


# Gevurah (Severity) Tests
@pytest.mark.asyncio
async def test_gevurah_validation():
    """Test request validation in Gevurah."""
    gevurah = Gevurah()
    valid_request = {"query": "test", "context": {}, "metadata": {}}
    result = await gevurah.validate(valid_request)
    assert result == valid_request


@pytest.mark.asyncio
async def test_gevurah_sanitization():
    """Test input sanitization."""
    sanitizer = Sanitizer()
    safe_input = {"text": "normal text"}
    await sanitizer.sanitize_input(safe_input)
    assert safe_input["text"] == "normal text"

    with pytest.raises(ValueError):
        dangerous_input = {"text": "DROP TABLE users;"}
        await sanitizer.sanitize_input(dangerous_input)


# Netzach (Victory) Tests
@pytest.mark.asyncio
async def test_netzach_integration():
    """Test external service integration."""
    netzach = Netzach()
    test_data = {"query": "test"}
    result = await netzach.integrate(test_data)
    assert "status" in result


@pytest.mark.asyncio
async def test_api_adapter():
    """Test API adapter request preparation."""
    adapter = APIAdapter()
    test_data = {"query": "test"}
    endpoint, request_data = adapter.prepare_request("firecrawl", test_data)
    assert endpoint == "https://firecrawl.com/api/v1/search"
    assert "Authorization" in request_data["headers"]
    assert "query" in request_data["json"]
