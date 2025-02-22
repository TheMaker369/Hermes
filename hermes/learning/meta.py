"""
Meta-learning system combining multiple learning paradigms.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ..core.memory import memory_manager
from ..quantum.optimization import optimizer
from .reinforcement import QuantumPPOAgent


@dataclass
class MetaLearningState:
    """State of the meta-learning system."""

    performance: Dict[str, float]
    active_paradigm: str
    learning_rate: float
    adaptation_rate: float


class HybridLearningSystem:
    """Combines multiple learning paradigms with meta-learning."""

    def __init__(self):
        """Initialize hybrid learning system."""
        self.state = MetaLearningState(
            performance={
                "reinforcement": 0.0,
                "supervised": 0.0,
                "unsupervised": 0.0,
                "quantum": 0.0,
            },
            active_paradigm="reinforcement",
            learning_rate=0.001,
            adaptation_rate=0.1,
        )

        # Initialize learning components
        self.rl_agent = QuantumPPOAgent(state_dim=64, action_dim=10)
        self.quantum_optimizer = optimizer.quantum_neural_network(
            n_qubits=8, n_layers=2
        )

        # Memory systems
        self.memory = memory_manager.create_collection("meta_learning")

    def learn(
        self, data: Union[Dict[str, Any], np.ndarray], paradigm: Optional[str] = None
    ) -> Dict[str, Any]:
        """Learn from data using the most appropriate paradigm."""
        if paradigm is None:
            paradigm = self._select_best_paradigm(data)

        result = None

        if paradigm == "reinforcement":
            result = self._reinforcement_learning(data)
        elif paradigm == "supervised":
            result = self._supervised_learning(data)
        elif paradigm == "unsupervised":
            result = self._unsupervised_learning(data)
        elif paradigm == "quantum":
            result = self._quantum_learning(data)

        # Update performance metrics
        self._update_performance(paradigm, result)

        # Store learning experience
        self._store_experience(paradigm, data, result)

        return result

    def _select_best_paradigm(self, data: Any) -> str:
        """Select the best learning paradigm for the data."""
        # Analyze data characteristics
        if isinstance(data, dict) and "reward" in data:
            return "reinforcement"
        elif isinstance(data, dict) and "label" in data:
            return "supervised"
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            return "unsupervised"
        else:
            return "quantum"

    def _reinforcement_learning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reinforcement learning."""
        state = data.get("state")
        reward = data.get("reward")

        if state is None or reward is None:
            return {"error": "Invalid RL data"}

        action, log_prob = self.rl_agent.select_action(state)
        self.rl_agent.store_experience(Experience(state, action, reward, state, False))

        return self.rl_agent.learn()

    def _supervised_learning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply supervised learning with quantum enhancement."""
        inputs = data.get("inputs")
        labels = data.get("labels")

        if inputs is None or labels is None:
            return {"error": "Invalid supervised data"}

        # Quantum feature extraction
        quantum_features = self.quantum_optimizer(self._get_random_params(), inputs)

        # Classical supervised learning
        criterion = nn.CrossEntropyLoss()
        outputs = torch.tensor(quantum_features)
        labels = torch.tensor(labels)
        loss = criterion(outputs, labels)

        return {"loss": loss.item()}

    def _unsupervised_learning(self, data: np.ndarray) -> Dict[str, Any]:
        """Apply unsupervised learning with quantum clustering."""
        # Quantum dimensionality reduction
        quantum_state = self.quantum_optimizer(self._get_random_params(), data)

        # Clustering in quantum feature space
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=min(8, len(data)))
        clusters = kmeans.fit_predict(quantum_state)

        return {"clusters": clusters, "inertia": kmeans.inertia_}

    def _quantum_learning(self, data: Any) -> Dict[str, Any]:
        """Apply pure quantum learning."""
        # Convert data to quantum state
        quantum_data = np.array(data).flatten()
        quantum_state = self.quantum_optimizer(self._get_random_params(), quantum_data)

        # Optimize quantum state
        optimized = optimizer.quantum_gradient_descent(
            optimizer.create_qaoa_circuit(len(quantum_state)), lambda x: np.sum(x**2)
        )

        return {"quantum_state": quantum_state, "optimization": optimized}

    def _update_performance(self, paradigm: str, result: Dict[str, Any]) -> None:
        """Update performance metrics for learning paradigms."""
        if "error" in result:
            self.state.performance[paradigm] *= 0.9  # Decay on error
        else:
            # Update based on result metrics
            if paradigm == "reinforcement":
                self.state.performance[paradigm] = result.get("mean_reward", 0.0)
            elif paradigm == "supervised":
                self.state.performance[paradigm] = 1.0 - result.get("loss", 1.0)
            elif paradigm == "unsupervised":
                self.state.performance[paradigm] = 1.0 / (
                    1.0 + result.get("inertia", 0.0)
                )
            elif paradigm == "quantum":
                self.state.performance[paradigm] = result.get("optimization", {}).get(
                    "optimal_value", 0.0
                )

    def _store_experience(
        self, paradigm: str, data: Any, result: Dict[str, Any]
    ) -> None:
        """Store learning experience in memory."""
        self.memory.add(
            documents=[str(data)],
            metadatas=[
                {
                    "paradigm": paradigm,
                    "performance": self.state.performance[paradigm],
                    "result": str(result),
                }
            ],
            ids=[str(len(self.memory.get()["ids"]) + 1)],
        )

    def _get_random_params(self) -> np.ndarray:
        """Get random parameters for quantum circuit."""
        return np.random.random(48)  # 8 qubits * 2 layers * 3 rotations

    def adapt_learning_rate(self) -> None:
        """Adapt learning rate based on performance."""
        mean_performance = np.mean(list(self.state.performance.values()))
        if mean_performance > 0.8:
            self.state.learning_rate *= 0.9  # Reduce learning rate when performing well
        else:
            self.state.learning_rate *= 1.1  # Increase learning rate when struggling

    def get_state(self) -> MetaLearningState:
        """Get current state of the learning system."""
        return self.state

    def save_state(self, path: str) -> None:
        """Save learning system state."""
        state_dict = {
            "meta_state": self.state,
            "rl_agent": self.rl_agent.save(path + "_rl"),
            "performance": self.state.performance,
        }
        torch.save(state_dict, path)

    def load_state(self, path: str) -> None:
        """Load learning system state."""
        state_dict = torch.load(path)
        self.state = state_dict["meta_state"]
        self.rl_agent.load(path + "_rl")
        self.state.performance = state_dict["performance"]


# Initialize hybrid learning system
learning_system = HybridLearningSystem()
