"""
Practical demonstration of Hermes quantum capabilities.
"""

from hermes.quantum.processor import quantum_processor
from hermes.quantum.optimization import optimizer
from hermes.quantum.security import security
import numpy as np

def demonstrate_practical_quantum():
    """Show what's practically possible today."""
    
    # 1. Quantum-Inspired Pattern Recognition
    print("1. Pattern Recognition")
    pattern = [0.1, 0.2, 0.3, 0.4]
    result = quantum_processor.process_pattern(pattern)
    print(f"Pattern coherence: {result['coherence']:.2f}")
    
    # 2. Quantum-Safe Security (Works Today)
    print("\n2. Quantum-Safe Security")
    message = b"Hello Quantum World"
    encrypted = security.quantum_safe_encryption(message)
    print(f"Encrypted data length: {len(encrypted['encrypted'])}")
    
    # 3. Optimization (Quantum-Inspired but Practical)
    print("\n3. Practical Optimization")
    def simple_cost(params):
        return np.sum(params**2)  # Simple quadratic cost
        
    result = optimizer.quantum_gradient_descent(
        quantum_processor.create_tree_circuit([0.1, 0.2]),
        simple_cost,
        initial_params=np.array([0.5, 0.5])
    )
    print(f"Optimization result: {result['optimal_value']:.2f}")
    
    # 4. Quantum Random Numbers (Simulated but Useful)
    print("\n4. Quantum Random Numbers")
    random_bits = security.generate_quantum_random(10)
    print(f"Random bits: {random_bits}")
    
    # 5. Error Mitigation (Practical Today)
    print("\n5. Error Mitigation")
    circuit = quantum_processor.create_tree_circuit([0.1, 0.2])
    mitigated = quantum_processor.apply_quantum_transform(
        np.array([0.1, 0.2])
    )
    print(f"Error-mitigated output shape: {mitigated.shape}")

if __name__ == "__main__":
    demonstrate_practical_quantum()
