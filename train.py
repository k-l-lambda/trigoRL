"""
Main training entry point for TrigoRL.

Usage:
    python train.py                          # Use default config
    python train.py agent=random env=dummy   # Override specific configs
    python train.py training.n_episodes=500  # Override parameters
"""

import os
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from trigor.agents.registry import make_agent
from trigor.envs.registry import make_env
from trigor.training.rl_trainer import RLTrainer


# Load environment variables
load_dotenv()


def set_seed(seed: int) -> None:
	"""Set random seeds for reproducibility."""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	import numpy as np
	import random

	np.random.seed(seed)
	random.seed(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
	"""
	Main training function.

	Args:
	    config: Hydra configuration
	"""
	# Print configuration
	print("=" * 80)
	print("TrigoRL Training")
	print("=" * 80)
	print("\nConfiguration:")
	print(OmegaConf.to_yaml(config))
	print("=" * 80)

	# Set seed
	set_seed(config.seed)

	# Create environment
	print(f"\nCreating environment: {config.env.type}")
	env = make_env(env_type=config.env.type, config=dict(config.env))
	print(f"  Observation space: {env.observation_space}")
	print(f"  Action space: {env.action_space}")

	# Create agent
	print(f"\nCreating agent: {config.agent.type}")
	agent = make_agent(
		agent_type=config.agent.type,
		observation_space=env.observation_space,
		action_space=env.action_space,
		config=dict(config.agent.params) if 'params' in config.agent else {},
	)

	# Create trainer
	print(f"\nInitializing trainer...")
	trainer = RLTrainer(config=config, agent=agent, env=env)

	# Start training
	print(f"\nStarting training...")
	trainer.train()


if __name__ == "__main__":
	main()
