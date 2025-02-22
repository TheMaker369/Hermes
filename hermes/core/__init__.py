"""
Core module initialization for Hermes AI system.
"""

from .harmonics import FrequencyHarmonizer
from .orchestrator import KeterOrchestrator
from .quantum import QuantumProcessor

Hermes = KeterOrchestrator  # Main interface alias

__all__ = ["Hermes", "QuantumProcessor", "FrequencyHarmonizer"]
