"""
Test suite for consciousness integration system.
"""

import pytest
import numpy as np
import torch
from hermes.consciousness.field_detector import ConsciousnessFieldDetector
from hermes.consciousness.harmonics import SacredHarmonics
from hermes.consciousness.integrator import QuantumConsciousnessIntegrator

def test_field_detector():
    """Test consciousness field detection."""
    detector = ConsciousnessFieldDetector()
    
    # Test field detection
    resonances = detector.detect_field(0.1)
    assert len(resonances) > 0
    
    # Test field analysis
    metrics = detector.analyze_coherence(resonances)
    assert 0 <= metrics['average_coherence'] <= 1
    
    # Test quantum correction
    state = np.array([1, 0], dtype=np.complex128)
    corrected = detector.apply_consciousness_correction(state, resonances)
    assert len(corrected) == 2
    assert abs(np.linalg.norm(corrected) - 1.0) < 1e-6

def test_sacred_harmonics():
    """Test sacred geometry harmonics."""
    harmonics = SacredHarmonics()
    
    # Test pattern setup
    assert 'merkaba' in harmonics.patterns
    assert 'sri_yantra' in harmonics.patterns
    
    # Test resonance calculation
    pattern = harmonics.patterns['merkaba']
    resonance = harmonics.calculate_resonance(pattern, 0.0)
    assert isinstance(resonance, complex)
    
    # Test consciousness integration
    state = np.array([1, 0], dtype=np.complex128)
    transformed = harmonics.integrate_consciousness(state, ['merkaba'])
    assert len(transformed) == 2
    assert abs(np.linalg.norm(transformed) - 1.0) < 1e-6

@pytest.mark.gpu
def test_quantum_consciousness_integration():
    """Test quantum consciousness integration."""
    integrator = QuantumConsciousnessIntegrator()
    
    # Test consciousness evolution
    initial_state = np.array([1, 0], dtype=np.complex128)
    states = integrator.evolve_consciousness(initial_state, 1.0, 10)
    assert len(states) == 10
    
    # Test evolution analysis
    metrics = integrator.analyze_consciousness_evolution(states)
    assert 0 <= metrics['average_coherence'] <= 1
    assert 0 <= metrics['average_fidelity'] <= 1
    
    # Test consciousness transformation
    transformed = integrator.apply_consciousness_transformation(
        initial_state, 'merkaba'
    )
    assert len(transformed) == 2
    assert abs(np.linalg.norm(transformed) - 1.0) < 1e-6

def test_consciousness_field_generation():
    """Test consciousness field generation and visualization."""
    integrator = QuantumConsciousnessIntegrator()
    
    # Generate initial state
    initial_state = np.array([1, 0], dtype=np.complex128)
    states = integrator.evolve_consciousness(initial_state, 0.1, 1)
    
    # Generate field
    field = integrator.generate_consciousness_field(states[0])
    assert field.shape == (32, 32, 32)
    assert np.iscomplexobj(field)
    
    # Test field properties
    assert np.all(np.isfinite(field))
    assert np.any(np.abs(field) > 0)

@pytest.mark.benchmark
def test_consciousness_performance(benchmark):
    """Benchmark consciousness integration performance."""
    integrator = QuantumConsciousnessIntegrator()
    initial_state = np.array([1, 0], dtype=np.complex128)
    
    def evolution_step():
        states = integrator.evolve_consciousness(initial_state, 0.1, 1)
        integrator.analyze_consciousness_evolution(states)
        
    # Run benchmark
    result = benchmark(evolution_step)
    assert result.stats.mean < 0.1  # Should complete in under 100ms

def test_consciousness_stability():
    """Test stability of consciousness evolution."""
    integrator = QuantumConsciousnessIntegrator()
    initial_state = np.array([1, 0], dtype=np.complex128)
    
    # Evolve for multiple steps
    states = integrator.evolve_consciousness(initial_state, 1.0, 100)
    
    # Check stability metrics
    metrics = integrator.analyze_consciousness_evolution(states)
    assert metrics['state_stability'] > 0.5  # Should be relatively stable
    assert metrics['coherence_stability'] > 0.5
    
    # Check conservation of probability
    for state in states:
        assert abs(np.linalg.norm(state.quantum_state) - 1.0) < 1e-6

@pytest.mark.integration
def test_full_consciousness_pipeline():
    """Test full consciousness integration pipeline."""
    # Initialize components
    detector = ConsciousnessFieldDetector()
    harmonics = SacredHarmonics()
    integrator = QuantumConsciousnessIntegrator()
    
    # Initial state
    initial_state = np.array([1, 0], dtype=np.complex128)
    
    # Evolution
    states = integrator.evolve_consciousness(initial_state, 1.0, 10)
    
    # Analysis
    metrics = integrator.analyze_consciousness_evolution(states)
    
    # Field generation
    field = integrator.generate_consciousness_field(states[-1])
    
    # Verify results
    assert len(states) == 10
    assert all(0 <= v <= 1 for v in metrics.values())
    assert field.shape == (32, 32, 32)
    
    # Check statistics
    stats = integrator.get_evolution_statistics()
    assert 'coherence_evolution' in stats
    assert 'field_statistics' in stats
