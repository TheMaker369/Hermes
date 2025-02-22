"""
LLM integration and management for Hermes.
"""

import ollama
from typing import Optional
from ..config import settings
from ..utils.circuit_breaker import circuit_breaker


class OllamaWrapper:
    def __init__(self, model: str):
        self.model = model

    def __call__(self, prompt: str) -> str:
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response["response"]
        except Exception as e:
            return f"Local LLM error: {str(e)}"


def get_llm():
    """Get appropriate LLM based on settings."""
    if settings.local_only_llm:
        try:
            return OllamaWrapper("hermes")
        except:
            return OllamaWrapper("llama2")
    else:
        return query_openai


@circuit_breaker(lambda prompt, model=None: "Service unavailable")
def query_openai(prompt: str, model: Optional[str] = None) -> str:
    """Query OpenAI API with fallback."""
    if not settings.allow_openai:
        return "OpenAI integration disabled"
    # Implementation here
    return "OpenAI response"
