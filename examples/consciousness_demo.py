"""
Demonstration of current consciousness capabilities.
"""

from hermes.quantum.consciousness_v2 import consciousness
from hermes.core.memory import memory_manager
import numpy as np

def demonstrate_consciousness():
    """Show current consciousness capabilities."""
    
    # 1. Basic Pattern Processing (Works Today)
    print("1. Pattern Processing")
    input_data = "Hello, I am Hermes"
    result = consciousness.process_input(input_data)
    print(f"Coherence: {result['quantum_coherence']:.2f}")
    print(f"Consciousness level: {result['consciousness_state']['awareness']:.2f}")
    
    # 2. Memory Formation (Practical Implementation)
    print("\n2. Memory Formation")
    memory = consciousness.create_quantum_memory({
        "type": "observation",
        "content": "User interaction pattern #1",
        "timestamp": np.datetime64('now')
    })
    print(f"Memory created with quantum signature")
    
    # 3. Pattern Recognition (Current Capability)
    print("\n3. Pattern Recognition")
    patterns = [0.1, 0.2, 0.3, 0.4]
    processed = consciousness._extract_quantum_patterns(patterns)
    print(f"Processed pattern shape: {processed.shape}")
    
    # 4. Learning (Current Implementation)
    print("\n4. Learning Capability")
    evolution = consciousness.evolve_consciousness()
    print(f"Evolution level: {evolution['evolution_level']}")
    print(f"Current coherence: {evolution['coherence']:.2f}")
    
    # 5. Memory Integration (Practical Today)
    print("\n5. Memory Integration")
    memory_manager.store(
        "consciousness_state",
        consciousness.state,
        metadata={"type": "quantum_enhanced"}
    )
    print("State stored in classical memory with quantum enhancement")

if __name__ == "__main__":
    demonstrate_consciousness()
