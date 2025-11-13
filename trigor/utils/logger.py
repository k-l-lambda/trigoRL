"""Weights & Biases logger for TrigoRL."""

import logging
import os
from typing import Any, Dict, Optional

import wandb


logger = logging.getLogger(__name__)


class WandbLogger:
	"""
	Wandb logger for experiment tracking.

	Handles initialization, metric logging, and artifact management.
	"""

	def __init__(
		self,
		project: str,
		entity: Optional[str] = None,
		name: Optional[str] = None,
		config: Optional[Dict[str, Any]] = None,
		tags: Optional[list] = None,
		enabled: bool = True,
	):
		"""
		Initialize wandb logger.

		Args:
		    project: Wandb project name
		    entity: Wandb entity (username or team)
		    name: Experiment name
		    config: Configuration dictionary to log
		    tags: Experiment tags
		    enabled: If False, disable wandb logging
		"""
		self.enabled = enabled

		if not self.enabled:
			print("Wandb logging disabled")
			return

		# Get entity from env if not provided
		if entity is None:
			entity = os.getenv('WANDB_ENTITY')

		# Initialize wandb
		self.run = wandb.init(
			project=project,
			entity=entity,
			name=name,
			config=config,
			tags=tags or [],
			reinit=True,
		)

		print(f"Wandb initialized: {self.run.url}")

	def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
		"""
		Log metrics to wandb.

		Args:
		    metrics: Dictionary of metric names and values
		    step: Global step number
		"""
		if not self.enabled:
			return

		wandb.log(metrics, step=step)

	def log_config(self, config: Dict[str, Any]) -> None:
		"""
		Update wandb config.

		Args:
		    config: Configuration dictionary
		"""
		if not self.enabled:
			return

		wandb.config.update(config)

	def save_checkpoint(self, checkpoint_path: str) -> None:
		"""
		Save checkpoint as wandb artifact.

		Args:
		    checkpoint_path: Path to checkpoint file
		"""
		if not self.enabled:
			return

		try:
			artifact = wandb.Artifact(name='model', type='model')
			artifact.add_file(checkpoint_path)
			wandb.log_artifact(artifact)
			logger.info(f"Successfully uploaded checkpoint artifact: {checkpoint_path}")
		except Exception as e:
			logger.warning(f"Failed to upload checkpoint artifact to wandb: {e}")
			logger.warning("Training will continue without artifact upload")

	def watch_model(self, model, log: str = 'all', log_freq: int = 100) -> None:
		"""
		Watch PyTorch model gradients and parameters.

		Args:
		    model: PyTorch model
		    log: What to log ('gradients', 'parameters', 'all')
		    log_freq: Logging frequency
		"""
		if not self.enabled:
			return

		wandb.watch(model, log=log, log_freq=log_freq)

	def finish(self) -> None:
		"""Finish wandb run."""
		if not self.enabled:
			return

		wandb.finish()

	def __enter__(self):
		"""Context manager entry."""
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
		"""Context manager exit."""
		self.finish()
