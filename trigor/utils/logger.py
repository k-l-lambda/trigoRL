"""Weights & Biases and TensorBoard loggers for TrigoRL."""

import logging
import os
from pathlib import Path
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
		run_id: Optional[str] = None,
		resume: Optional[str] = None,
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
		    run_id: Wandb run ID to resume (if resuming existing run)
		    resume: Resume mode ('allow', 'must', 'never', or None)
		"""
		self.enabled = enabled

		if not self.enabled:
			print("Wandb logging disabled")
			return

		# Get entity from env if not provided
		if entity is None:
			entity = os.getenv('WANDB_ENTITY')

		# Initialize wandb with optional resume support
		init_kwargs = {
			'project': project,
			'entity': entity,
			'name': name,
			'config': config,
			'tags': tags or [],
			'reinit': True,
		}

		# Add resume parameters if provided
		if run_id is not None:
			init_kwargs['id'] = run_id
			init_kwargs['resume'] = resume or 'allow'
			logger.info(f"Resuming wandb run: {run_id} (mode: {init_kwargs['resume']})")

		self.run = wandb.init(**init_kwargs)

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

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Context manager exit."""
		# Unused parameters required by context manager protocol
		_ = exc_type, exc_val, exc_tb
		self.finish()


class TensorBoardLogger:
	"""
	TensorBoard logger for experiment tracking.

	Handles initialization, metric logging, and histogram tracking.
	"""

	def __init__(
		self,
		log_dir: str,
		enabled: bool = True,
	):
		"""
		Initialize TensorBoard logger.

		Args:
		    log_dir: Directory to save TensorBoard logs
		    enabled: If False, disable TensorBoard logging
		"""
		self.enabled = enabled
		self.writer = None

		if not self.enabled:
			print("TensorBoard logging disabled")
			return

		try:
			from torch.utils.tensorboard import SummaryWriter

			# Create log directory if it doesn't exist
			log_path = Path(log_dir)
			log_path.mkdir(parents=True, exist_ok=True)

			# Initialize TensorBoard writer
			self.writer = SummaryWriter(log_dir=str(log_path))
			print(f"TensorBoard initialized: {log_path}")
			print(f"  View with: tensorboard --logdir={log_path}")

		except ImportError:
			logger.warning("torch.utils.tensorboard not available, disabling TensorBoard logging")
			self.enabled = False
		except Exception as e:
			logger.warning(f"Failed to initialize TensorBoard: {e}")
			self.enabled = False

	def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
		"""
		Log metrics to TensorBoard.

		Args:
		    metrics: Dictionary of metric names and values
		    step: Global step number
		"""
		if not self.enabled or self.writer is None:
			return

		for key, value in metrics.items():
			if isinstance(value, (int, float)):
				self.writer.add_scalar(key, value, global_step=step)

	def log_config(self, config: Dict[str, Any]) -> None:
		"""
		Log configuration as text to TensorBoard.

		Args:
		    config: Configuration dictionary
		"""
		if not self.enabled or self.writer is None:
			return

		# Convert config to text
		import yaml
		config_text = yaml.dump(config, default_flow_style=False)
		self.writer.add_text('config', config_text, global_step=0)

	def log_histogram(self, tag: str, values, step: Optional[int] = None) -> None:
		"""
		Log histogram to TensorBoard.

		Args:
		    tag: Name of the histogram
		    values: Values to plot (numpy array or torch tensor)
		    step: Global step number
		"""
		if not self.enabled or self.writer is None:
			return

		self.writer.add_histogram(tag, values, global_step=step)

	def watch_model(self, model, log_freq: int = 100) -> None:
		"""
		Watch PyTorch model gradients and parameters.

		Note: This is a placeholder for compatibility with WandbLogger.
		For TensorBoard, gradients need to be logged manually in training loop.

		Args:
		    model: PyTorch model
		    log_freq: Logging frequency (not used)
		"""
		if not self.enabled:
			return

		# TensorBoard doesn't have automatic model watching like wandb
		# Gradients need to be logged manually
		logger.info("TensorBoard: Model watching requires manual gradient logging")

	def flush(self) -> None:
		"""Flush pending logs to disk."""
		if not self.enabled or self.writer is None:
			return

		self.writer.flush()

	def finish(self) -> None:
		"""Close TensorBoard writer."""
		if not self.enabled or self.writer is None:
			return

		self.writer.close()
		logger.info("TensorBoard writer closed")

	def __enter__(self):
		"""Context manager entry."""
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Context manager exit."""
		# Unused parameters required by context manager protocol
		_ = exc_type, exc_val, exc_tb
		self.finish()
