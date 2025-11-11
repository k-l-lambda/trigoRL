"""Base environment wrapper for TrigoRL."""

from abc import ABC
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class BaseEnv(ABC):
    """
    Base environment wrapper for TrigoRL.

    Wraps gymnasium environments with a consistent interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the environment.

        Args:
            config: Environment-specific configuration
        """
        self.config = config or {}
        self.env = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        return self.env.reset(seed=seed)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return transition.

        Returns:
            observation, reward, terminated, truncated, info
        """
        return self.env.step(action)

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            self.env.close()

    @property
    def observation_space(self):
        """Return observation space."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Return action space."""
        return self.env.action_space


class DummyEnv(BaseEnv):
    """
    Simple dummy environment for testing.

    Provides random observations and rewards for framework validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Get parameters from config
        params = self.config.get('params', {})
        obs_dim = params.get('observation_dim', 10)
        act_dim = params.get('action_dim', 4)
        self.episode_length = params.get('episode_length', 100)
        self.reward_range = params.get('reward_range', [-1.0, 1.0])

        # Create simple spaces
        self._observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(act_dim)

        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset to random initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        obs = self._observation_space.sample()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Return random observation and reward."""
        self.step_count += 1

        obs = self._observation_space.sample()
        reward = np.random.uniform(*self.reward_range)
        terminated = self.step_count >= self.episode_length
        truncated = False

        info = {'step': self.step_count}

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Nothing to close."""
        pass

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
