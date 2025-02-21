"""
Core module initialization for Hermes AI system.
"""

from .orchestrator import KeterOrchestrator
from .quantum import QuantumProcessor
from .harmonics import FrequencyHarmonizer

Hermes = KeterOrchestrator  # Main interface alias

__all__ = ['Hermes', 'QuantumProcessor', 'FrequencyHarmonizer']
