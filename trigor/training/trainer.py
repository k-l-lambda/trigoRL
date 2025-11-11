"""Custom RL trainer for TrigoRL.

Inspired by deep-starry's trainer but adapted for reinforcement learning.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from trigor.agents.base import BaseAgent
from trigor.envs.base import BaseEnv
from trigor.utils.checkpoint import CheckpointManager
from trigor.utils.logger import WandbLogger


class RLTrainer:
    """
    Custom RL trainer for episode-based training.

    Handles:
    - Episode collection and evaluation
    - Checkpoint saving/loading
    - Metric logging to wandb
    - Best model tracking
    """

    def __init__(
        self,
        config: DictConfig,
        agent: BaseAgent,
        env: BaseEnv,
        eval_env: Optional[BaseEnv] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Full Hydra configuration
            agent: RL agent
            env: Training environment
            eval_env: Evaluation environment (optional, uses env if None)
        """
        self.config = config
        self.agent = agent
        self.env = env
        self.eval_env = eval_env or env

        # Training config
        self.n_episodes = config.training.n_episodes
        self.max_steps = config.training.max_steps_per_episode
        self.eval_frequency = config.training.eval_frequency
        self.eval_episodes = config.training.eval_episodes

        # Device
        self.device = config.device

        # Initialize logger
        self.logger = WandbLogger(
            project=config.experiment.project,
            name=config.experiment.name,
            config=dict(config),
            tags=config.experiment.tags,
            enabled=config.training.log.wandb,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config.paths.output) / 'checkpoints',
            save_mode=config.training.save_mode,
            monitor_field=config.training.monitor.field,
            monitor_mode=config.training.monitor.mode,
            keep_n_checkpoints=config.training.keep_n_checkpoints,
        )

        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.best_eval_metric = float('-inf') if config.training.monitor.mode == 'max' else float('inf')

        # Try to load checkpoint
        self._try_load_checkpoint()

    def _try_load_checkpoint(self) -> None:
        """Try to load existing checkpoint if available."""
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()

        if latest_checkpoint is not None:
            try:
                checkpoint = self.checkpoint_manager.load(
                    checkpoint_name='latest.chkpt',
                    device=self.device,
                )

                self.current_episode = checkpoint.get('episode', 0)
                self.total_steps = checkpoint.get('total_steps', 0)
                self.best_eval_metric = checkpoint.get('best_eval_metric', self.best_eval_metric)

                # Load agent state if available
                if 'agent_state' in checkpoint:
                    # Agent should implement load_state_dict or similar
                    print(f"Loaded checkpoint from episode {self.current_episode}")
                else:
                    print("Checkpoint found but no agent state")

            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    def train(self) -> None:
        """Main training loop."""
        print(f"\nStarting training from episode {self.current_episode}")
        print(f"Target episodes: {self.n_episodes}")
        print(f"Device: {self.device}\n")

        while self.current_episode < self.n_episodes:
            # Run training episode
            train_metrics = self._run_episode(training=True)

            # Log training metrics
            self._log_metrics(train_metrics, prefix='train')

            # Evaluate
            if (self.current_episode + 1) % self.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self._log_metrics(eval_metrics, prefix='eval')

                # Save checkpoint
                self._save_checkpoint(eval_metrics)

            self.current_episode += 1

        print("\nTraining completed!")
        self.logger.finish()

    def _run_episode(self, training: bool = True) -> Dict[str, float]:
        """
        Run a single episode.

        Args:
            training: If True, run training episode; if False, evaluation

        Returns:
            Episode metrics
        """
        obs, info = self.env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0

        while not done and step_count < self.max_steps:
            # Select action
            action = self.agent.act(obs, deterministic=not training)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step_count += 1
            self.total_steps += 1

            obs = next_obs

        metrics = {
            'episode_reward': episode_reward,
            'episode_length': step_count,
        }

        return metrics

    def _evaluate(self) -> Dict[str, float]:
        """
        Run evaluation episodes.

        Returns:
            Average evaluation metrics
        """
        print(f"\nEvaluating at episode {self.current_episode}...")

        rewards = []
        lengths = []

        for _ in tqdm(range(self.eval_episodes), desc='Eval'):
            metrics = self._run_episode(training=False)
            rewards.append(metrics['episode_reward'])
            lengths.append(metrics['episode_length'])

        eval_metrics = {
            'episode_reward': np.mean(rewards),
            'episode_reward_std': np.std(rewards),
            'episode_length': np.mean(lengths),
        }

        print(
            f"Eval Results: Reward={eval_metrics['episode_reward']:.2f} ± "
            f"{eval_metrics['episode_reward_std']:.2f}, "
            f"Length={eval_metrics['episode_length']:.1f}"
        )

        return eval_metrics

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = '') -> None:
        """
        Log metrics to wandb and console.

        Args:
            metrics: Metrics dictionary
            prefix: Prefix for metric names (e.g., 'train', 'eval')
        """
        # Add prefix
        if prefix:
            metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}

        # Add episode number
        metrics['episode'] = self.current_episode
        metrics['total_steps'] = self.total_steps

        # Log to wandb
        self.logger.log(metrics, step=self.current_episode)

        # Log to console
        if self.config.training.log.console:
            if (self.current_episode + 1) % self.config.training.log.frequency == 0:
                metric_str = ', '.join([f'{k}={v:.3f}' for k, v in metrics.items() if k not in ['episode', 'total_steps']])
                print(f"Episode {self.current_episode}: {metric_str}")

    def _save_checkpoint(self, eval_metrics: Dict[str, float]) -> None:
        """
        Save training checkpoint.

        Args:
            eval_metrics: Evaluation metrics for determining best model
        """
        checkpoint = {
            'episode': self.current_episode,
            'total_steps': self.total_steps,
            'best_eval_metric': self.best_eval_metric,
            'config': dict(self.config),
        }

        # Get monitored metric value
        monitor_field = self.config.training.monitor.field
        metric_value = eval_metrics.get(monitor_field)

        # Save checkpoint
        saved_path = self.checkpoint_manager.save(
            checkpoint=checkpoint,
            episode=self.current_episode,
            metric_value=metric_value,
            is_latest=True,
        )

        if saved_path:
            print(f"Saved checkpoint: {Path(saved_path).name}")

            # Upload to wandb
            if self.config.training.log.wandb:
                self.logger.save_checkpoint(saved_path)
