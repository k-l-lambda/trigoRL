"""Base agent interface for TrigoRL."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseAgent(ABC):
	"""
	Base class for all RL agents in TrigoRL.

	All agents must implement the act() method to select actions
	and optionally implement update() for learning.
	"""

	def __init__(self, observation_space, action_space, config: Optional[Dict[str, Any]] = None):
		"""
		Initialize the agent.

		Args:
		    observation_space: Environment observation space
		    action_space: Environment action space
		    config: Agent-specific configuration dictionary
		"""
		self.observation_space = observation_space
		self.action_space = action_space
		self.config = config or {}

	@abstractmethod
	def act(self, observation: np.ndarray, deterministic: bool = False) -> int:
		"""
		Select an action given an observation.

		Args:
		    observation: Current observation from environment
		    deterministic: If True, select best action; if False, sample from policy

		Returns:
		    Selected action (integer index)
		"""
		pass

	def update(self, *args, **kwargs) -> Dict[str, float]:
		"""
		Update agent parameters (for learning algorithms).

		Returns:
		    Dictionary of training metrics
		"""
		return {}

	def save(self, path: str) -> None:
		"""Save agent state to disk."""
		pass

	def load(self, path: str) -> None:
		"""Load agent state from disk."""
		pass

	def reset(self) -> None:
		"""Reset agent state (e.g., at episode start)."""
		pass


class RandomAgent(BaseAgent):
	"""
	Simple random agent for testing.

	Selects actions uniformly at random from the action space.
	"""

	def __init__(self, observation_space, action_space, config: Optional[Dict[str, Any]] = None):
		super().__init__(observation_space, action_space, config)

		# Set random seed if provided
		seed = self.config.get('seed')
		if seed is not None:
			np.random.seed(seed)

	def act(self, observation: np.ndarray, deterministic: bool = False) -> int:
		"""Select a random action."""
		return self.action_space.sample()
