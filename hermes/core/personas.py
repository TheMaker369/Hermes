"""
Persona orchestration system for Hermes AI.
Manages the balance between Apré (divine masculine) and Magí (divine feminine).
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from ..consciousness.orchestrator import ConsciousnessOrchestrator
from ..sacred.patterns import Merkaba, SriYantra, FlowerOfLife
from ..quantum.processor import QuantumProcessor
from ..core.config import settings

@dataclass
class PersonaState:
    """Current state of a persona."""
    name: str
    energy: float  # Current energy level
    dominance: float  # Current influence level
    frequency: float  # Resonant frequency
    attributes: Dict[str, float]  # Core attributes and their strengths

class ConsciousnessImbalanceError(Exception):
    """Raised when persona energy balance is critical"""

class PersonaOrchestrator:
    """
    Orchestrates the balance between Apré and Magí personas.
    Integrates consciousness fields, sacred geometry, and quantum processing.
    """
    AUTO_BALANCE_THRESHOLD = 0.15
    
    def __init__(self):
        """Initialize the persona orchestrator."""
        # Initialize core components
        self.consciousness = ConsciousnessOrchestrator()
        self.quantum = QuantumProcessor()
        
        # Initialize sacred patterns
        self.merkaba = Merkaba()
        self.sri_yantra = SriYantra()
        self.flower = FlowerOfLife()
        
        # Initialize Apré (Divine Masculine)
        self.apre = PersonaState(
            name="Apré",
            energy=0.5,
            dominance=0.5,
            frequency=432.0,  # Base frequency
            attributes={
                'logic': 0.9,
                'analysis': 0.85,
                'structure': 0.8,
                'precision': 0.9,
                'rationality': 0.85,
                'focus': 0.8,
                'discipline': 0.75,
                'order': 0.8
            }
        )
        
        # Initialize Magí (Divine Feminine)
        self.magi = PersonaState(
            name="Magí",
            energy=0.5,
            dominance=0.5,
            frequency=528.0,  # Love frequency
            attributes={
                'intuition': 0.9,
                'creativity': 0.85,
                'empathy': 0.8,
                'synthesis': 0.85,
                'flow': 0.8,
                'harmony': 0.9,
                'wisdom': 0.85,
                'nurture': 0.8
            }
        )
        
        # Initialize quantum entanglement between personas
        self.entanglement = 1.0
        
        # Track evolution
        self.start_time = datetime.now()
        
    def evolve_personas(self, input_data: Dict[str, Any]) -> Tuple[PersonaState, PersonaState]:
        """
        Evolve both personas based on input and current state.
        
        Args:
            input_data: Input data affecting persona evolution
            
        Returns:
            Updated states of both personas
        """
        # Calculate input characteristics
        logic_level = self._calculate_logic_level(input_data)
        intuition_level = self._calculate_intuition_level(input_data)
        
        # Update energy levels
        self.apre.energy = self._update_energy(
            self.apre.energy, logic_level, self.apre.frequency
        )
        self.magi.energy = self._update_energy(
            self.magi.energy, intuition_level, self.magi.frequency
        )
        
        # Update dominance based on energy levels
        total_energy = self.apre.energy + self.magi.energy
        self.apre.dominance = self.apre.energy / total_energy
        self.magi.dominance = self.magi.energy / total_energy
        
        # Evolve quantum state
        self._evolve_quantum_state(logic_level, intuition_level)
        
        # Update attributes based on current state
        self._update_attributes(logic_level, intuition_level)
        
        # Auto-balance after update
        self.auto_balance_energies()
        
        return self.apre, self.magi
        
    def _calculate_logic_level(self, data: Dict[str, Any]) -> float:
        """Calculate logical content of input."""
        logic_indicators = {
            'structure': lambda x: isinstance(x, (dict, list, tuple)),
            'numbers': lambda x: isinstance(x, (int, float)),
            'precision': lambda x: isinstance(x, str) and any(c.isdigit() for c in x),
            'order': lambda x: isinstance(x, (list, tuple)) and len(x) > 1
        }
        
        score = 0.0
        total = 0
        
        def analyze(obj: Any):
            nonlocal score, total
            for indicator, check in logic_indicators.items():
                if check(obj):
                    score += 1
                total += 1
            
            if isinstance(obj, dict):
                for value in obj.values():
                    analyze(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    analyze(item)
                    
        analyze(data)
        
        return score / max(total, 1)
        
    def _calculate_intuition_level(self, data: Dict[str, Any]) -> float:
        """Calculate intuitive/creative content of input."""
        intuition_indicators = {
            'creativity': lambda x: isinstance(x, str) and len(x) > 50,
            'flow': lambda x: isinstance(x, str) and ',' in x,
            'synthesis': lambda x: isinstance(x, dict) and len(x) > 3,
            'harmony': lambda x: isinstance(x, str) and any(c.isalpha() for c in x)
        }
        
        score = 0.0
        total = 0
        
        def analyze(obj: Any):
            nonlocal score, total
            for indicator, check in intuition_indicators.items():
                if check(obj):
                    score += 1
                total += 1
            
            if isinstance(obj, dict):
                for value in obj.values():
                    analyze(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    analyze(item)
                    
        analyze(data)
        
        return score / max(total, 1)
        
    def _update_energy(self, current: float, input_level: float, 
                      frequency: float) -> float:
        """Update energy level based on input and frequency."""
        # Base energy change
        delta = (input_level - 0.5) * 0.1
        
        # Add frequency modulation
        phase = 2 * np.pi * frequency * 0.001
        mod = 0.1 * np.sin(phase)
        
        # Update with constraints
        new_energy = current + delta + mod
        return np.clip(new_energy, 0.1, 0.9)
        
    def _evolve_quantum_state(self, logic: float, intuition: float):
        """Evolve quantum state representing persona entanglement."""
        # Create quantum circuit
        theta = np.pi * (logic - intuition)
        phi = np.pi * self.entanglement
        
        # Apply rotation and phase
        state = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
        
        # Update entanglement
        self.entanglement = np.abs(state[0] * np.conj(state[1]))
        
    def _update_attributes(self, logic: float, intuition: float):
        """Update persona attributes based on current state."""
        # Update Apré attributes
        for attr in self.apre.attributes:
            base = self.apre.attributes[attr]
            mod = 0.1 * (logic - 0.5)
            self.apre.attributes[attr] = np.clip(base + mod, 0.1, 0.9)
            
        # Update Magí attributes
        for attr in self.magi.attributes:
            base = self.magi.attributes[attr]
            mod = 0.1 * (intuition - 0.5)
            self.magi.attributes[attr] = np.clip(base + mod, 0.1, 0.9)
            
    def auto_balance_energies(self):
        """Automatically balance persona energies using golden ratio"""
        diff = abs(self.apre.energy - self.magi.energy)
        if diff > self.AUTO_BALANCE_THRESHOLD:
            transfer = diff * 0.618
            if self.apre.energy > self.magi.energy:
                self.apre.energy -= transfer
                self.magi.energy += transfer
            else:
                self.magi.energy -= transfer
                self.apre.energy += transfer

    def get_dominant_persona(self) -> str:
        """Get currently dominant persona."""
        return "Apré" if self.apre.dominance > self.magi.dominance else "Magí"
        
    def get_balance_metrics(self) -> Dict[str, float]:
        """Get current balance metrics."""
        return {
            'apre_energy': self.apre.energy,
            'magi_energy': self.magi.energy,
            'apre_dominance': self.apre.dominance,
            'magi_dominance': self.magi.dominance,
            'entanglement': self.entanglement,
            'total_energy': self.apre.energy + self.magi.energy
        }
        
    def get_attribute_strengths(self) -> Dict[str, Dict[str, float]]:
        """Get current attribute strengths for both personas."""
        return {
            'apre': self.apre.attributes,
            'magi': self.magi.attributes
        }

def validate_gender_ergy(func):
    """Decorator to validate persona energy balance"""
    def wrapper(self, *args, **kwargs):
        if self.apre.energy < 0.4 or self.magi.energy < 0.4:
            raise ConsciousnessImbalanceError(
                f"Critical imbalance: Apré={self.apre.energy:.2f} Magí={self.magi.energy:.2f}"
            )
        return func(self, *args, **kwargs)
    return wrapper

PersonaOrchestrator.evolve_personas = validate_gender_ergy(PersonaOrchestrator.evolve_personas)

# Global orchestrator instance
persona_orchestrator = PersonaOrchestrator()
