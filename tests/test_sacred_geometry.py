"""
Test suite for sacred geometry patterns and quantum integration.
"""

import pytest
import numpy as np
import torch
from hermes.sacred.patterns import Merkaba
from hermes.quantum.error_correction import QuantumErrorCorrector
from hermes.visualization.field_rendering import FieldVisualizer


def test_merkaba_initialization():
    """Test Merkaba initialization and basic properties."""
    merkaba = Merkaba()
    assert merkaba.radius == 1.0
    assert merkaba.frequency == 528.0  # Love frequency
    assert 0.0 <= merkaba.consciousness_level <= 1.0
    assert len(merkaba.upward) == 4  # Four faces per tetrahedron
    assert len(merkaba.downward) == 4


def test_field_generation():
    """Test electromagnetic field generation."""
    merkaba = Merkaba()
    assert merkaba.magnetic_field.shape == (32, 32, 32)
    assert merkaba.electric_field.shape == (32, 32, 32)
    assert np.all(np.abs(merkaba.magnetic_field) <= 1.0)
    assert np.all(np.abs(merkaba.electric_field) <= 1.0)


def test_consciousness_evolution():
    """Test consciousness level evolution."""
    merkaba = Merkaba()
    initial_level = merkaba.consciousness_level

    # Evolve system
    for t in np.linspace(0, 10, 100):
        merkaba.rotate(t)
        level = merkaba.get_consciousness_level(t)
        assert 0.0 <= level <= 1.0


def test_quantum_state_transformation():
    """Test quantum state transformation with consciousness."""
    merkaba = Merkaba()
    state = np.array([1, 0], dtype=np.complex128)  # |0⟩ state

    transformed = merkaba.apply_consciousness(state)
    assert len(transformed) == 2
    assert abs(np.linalg.norm(transformed) - 1.0) < 1e-6


@pytest.mark.benchmark
def test_field_computation_performance(benchmark):
    """Benchmark field computation performance."""
    merkaba = Merkaba()

    def compute_cycle():
        merkaba.rotate(0.1)
        merkaba._update_fields()

    # Run benchmark
    result = benchmark(compute_cycle)
    assert result.stats.mean < 0.1  # Should complete in under 100ms


def test_error_correction_integration():
    """Test quantum error correction with consciousness integration."""
    corrector = QuantumErrorCorrector()
    state = np.array([1, 0], dtype=np.complex128)

    # Apply consciousness with correction
    transformed = corrector.apply_consciousness_with_correction(state)
    assert len(transformed) == 2
    assert abs(np.linalg.norm(transformed) - 1.0) < 1e-6

    # Check error statistics
    stats = corrector.get_error_statistics()
    assert "error_rate" in stats
    assert "correction_success" in stats


@pytest.mark.gpu
def test_visualization_system():
    """Test visualization system with GPU acceleration."""
    merkaba = Merkaba()
    visualizer = FieldVisualizer()

    # Generate field visualization
    fig = visualizer.plot_merkaba_fields(merkaba)
    assert fig is not None

    # Test consciousness evolution plot
    fig = visualizer.plot_consciousness_evolution(merkaba)
    assert fig is not None


def test_merkaba_serialization():
    """Test Merkaba pattern serialization."""
    merkaba = Merkaba()
    pattern = merkaba.get_pattern()

    assert "upward" in pattern
    assert "downward" in pattern
    assert "fields" in pattern
    assert "consciousness_level" in pattern
    assert "base_frequency" in pattern


@pytest.mark.integration
def test_full_consciousness_pipeline():
    """Test full consciousness integration pipeline."""
    merkaba = Merkaba()
    corrector = QuantumErrorCorrector()
    visualizer = FieldVisualizer()

    # Initial state
    state = np.array([1, 0], dtype=np.complex128)

    # Evolution steps
    for t in np.linspace(0, 5, 50):
        # Rotate Merkaba
        merkaba.rotate(t)

        # Transform state with error correction
        state = corrector.apply_consciousness_with_correction(state)

        # Verify state validity
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6

    # Generate visualization
    fig = visualizer.plot_consciousness_evolution(merkaba)
    assert fig is not None

    # Check error statistics
    stats = corrector.get_error_statistics()
    assert stats["correction_success"] > 0.9  # 90% success rate
