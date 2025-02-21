"""
Advanced reinforcement learning with quantum enhancements.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from ..quantum.optimization import optimizer
from ..core.memory import memory_manager

@dataclass
class Experience:
    """Single reinforcement learning experience."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class QuantumPolicyNetwork(nn.Module):
    """Neural network with quantum-enhanced layers."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.quantum_circuit = optimizer.quantum_neural_network(
            n_qubits=min(state_dim, 8),
            n_layers=2
        )
        
        self.classical_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum processing for feature extraction
        quantum_features = torch.tensor(
            self.quantum_circuit(self._get_random_params(), x.numpy()),
            dtype=torch.float32
        )
        
        # Combine with classical processing
        classical_out = self.classical_net(x)
        combined = classical_out + quantum_features.reshape(-1, 1)
        
        return torch.softmax(combined, dim=-1)
        
    def _get_random_params(self) -> np.ndarray:
        """Get random parameters for quantum circuit."""
        return np.random.random(48)  # 8 qubits * 2 layers * 3 rotations

class QuantumAdvantageActor(nn.Module):
    """Actor network with quantum advantage."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.quantum_net = QuantumPolicyNetwork(state_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.quantum_net(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class QuantumCritic(nn.Module):
    """Critic network with quantum enhancement."""
    
    def __init__(self, state_dim: int):
        super().__init__()
        self.quantum_circuit = optimizer.quantum_neural_network(
            n_qubits=min(state_dim, 8),
            n_layers=2
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Quantum value estimation
        quantum_value = torch.tensor(
            self.quantum_circuit(self._get_random_params(), state.numpy()),
            dtype=torch.float32
        )
        
        # Classical value estimation
        classical_value = self.value_net(state)
        
        # Combine estimates
        return classical_value + quantum_value.mean()
        
    def _get_random_params(self) -> np.ndarray:
        return np.random.random(48)

class QuantumPPOAgent:
    """Quantum-enhanced Proximal Policy Optimization agent."""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = QuantumAdvantageActor(state_dim, action_dim)
        self.critic = QuantumCritic(state_dim)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Memory
        self.experiences = []
        self.quantum_memory = memory_manager.create_collection("quantum_experiences")
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using quantum-enhanced policy."""
        state_tensor = torch.FloatTensor(state)
        action, log_prob = self.actor(state_tensor)
        return action.item(), log_prob.item()
        
    def store_experience(self, experience: Experience) -> None:
        """Store experience in quantum-enhanced memory."""
        self.experiences.append(experience)
        
        # Store in quantum memory
        self.quantum_memory.add(
            documents=[str(experience)],
            metadatas=[{
                "reward": experience.reward,
                "done": experience.done
            }],
            ids=[str(len(self.experiences))]
        )
        
    def learn(self) -> Dict[str, float]:
        """Learn from stored experiences using PPO."""
        if len(self.experiences) < 128:  # Minimum batch size
            return {"loss": 0.0}
            
        # Prepare batches
        states = torch.FloatTensor([e.state for e in self.experiences])
        actions = torch.LongTensor([e.action for e in self.experiences])
        rewards = torch.FloatTensor([e.reward for e in self.experiences])
        next_states = torch.FloatTensor([e.next_state for e in self.experiences])
        dones = torch.FloatTensor([e.done for e in self.experiences])
        
        # Compute advantages
        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = self._compute_advantages(rewards, values, next_values, dones)
        
        # PPO update
        for _ in range(10):  # PPO epochs
            # Actor update
            _, new_log_probs = self.actor(states)
            ratio = torch.exp(new_log_probs - torch.tensor([e.log_prob for e in self.experiences]))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Critic update
            value_pred = self.critic(states)
            value_target = rewards + 0.99 * next_values * (1 - dones)
            critic_loss = nn.MSELoss()(value_pred, value_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
        # Clear experiences
        self.experiences = []
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_reward": rewards.mean().item()
        }
        
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> torch.Tensor:
        """Compute advantages using GAE."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.FloatTensor(advantages)
        
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
        
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

# Initialize agent with example dimensions
agent = QuantumPPOAgent(state_dim=64, action_dim=10)
