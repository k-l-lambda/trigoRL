"""Checkpoint management for TrigoRL."""

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
	"""
	Manage model checkpointing with best model tracking.

	Inspired by deep-starry's checkpoint system.
	"""

	def __init__(
		self,
		checkpoint_dir: str,
		save_mode: str = 'best',
		monitor_field: str = 'episode_reward',
		monitor_mode: str = 'max',
		keep_n_checkpoints: int = 5,
	):
		"""
		Initialize checkpoint manager.

		Args:
		    checkpoint_dir: Directory to save checkpoints
		    save_mode: 'best', 'all', or 'latest'
		    monitor_field: Metric name to monitor
		    monitor_mode: 'max' or 'min'
		    keep_n_checkpoints: Number of best checkpoints to keep
		"""
		self.checkpoint_dir = Path(checkpoint_dir)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

		self.save_mode = save_mode
		self.monitor_field = monitor_field
		self.monitor_mode = monitor_mode
		self.keep_n = keep_n_checkpoints

		# Track best metric value
		if monitor_mode == 'max':
			self.best_value = float('-inf')
		else:
			self.best_value = float('inf')

	def is_new_best(self, metric_value: float) -> bool:
		"""
		Check if metric value is new best.

		Args:
		    metric_value: Current metric value

		Returns:
		    True if new best
		"""
		if self.monitor_mode == 'max':
			is_best = metric_value > self.best_value
		else:
			is_best = metric_value < self.best_value

		if is_best:
			self.best_value = metric_value

		return is_best

	def save(
		self,
		checkpoint: Dict[str, Any],
		episode: int,
		metric_value: Optional[float] = None,
		is_latest: bool = True,
	) -> Optional[str]:
		"""
		Save checkpoint.

		Args:
		    checkpoint: Checkpoint dictionary
		    episode: Current episode number
		    metric_value: Metric value for naming
		    is_latest: If True, also save as latest.chkpt

		Returns:
		    Path to saved checkpoint, or None if not saved
		"""
		saved_path = None

		# Save latest checkpoint
		if is_latest:
			latest_path = self.checkpoint_dir / 'latest.chkpt'
			torch.save(checkpoint, latest_path)
			saved_path = str(latest_path)

		# Save based on mode
		if self.save_mode == 'all':
			# Save every checkpoint with episode number
			filename = f'checkpoint_ep{episode:04d}.chkpt'
			path = self.checkpoint_dir / filename
			torch.save(checkpoint, path)
			saved_path = str(path)

		elif self.save_mode == 'best' and metric_value is not None:
			# Save only if new best
			if self.is_new_best(metric_value):
				filename = f'best_ep{episode:04d}_{self.monitor_field}_{metric_value:.4f}.chkpt'
				path = self.checkpoint_dir / filename
				torch.save(checkpoint, path)
				saved_path = str(path)

				# Clean up old checkpoints
				self._cleanup_old_checkpoints()

		return saved_path

	def _cleanup_old_checkpoints(self) -> None:
		"""Remove old checkpoints, keeping only best N."""
		pattern = f'best_ep*_{self.monitor_field}_*.chkpt'
		checkpoints = glob.glob(str(self.checkpoint_dir / pattern))

		if len(checkpoints) <= self.keep_n:
			return

		# Sort by metric value (extract from filename)
		def extract_metric(path):
			filename = os.path.basename(path)
			# Format: best_ep0123_metric_0.1234.chkpt
			parts = filename.split('_')
			metric_str = parts[-1].replace('.chkpt', '')
			return float(metric_str)

		checkpoints.sort(key=extract_metric, reverse=(self.monitor_mode == 'max'))

		# Remove worst checkpoints
		for path in checkpoints[self.keep_n :]:
			os.remove(path)
			print(f"Removed old checkpoint: {os.path.basename(path)}")

	def load(self, checkpoint_name: str = 'latest.chkpt', device: str = 'cpu') -> Dict[str, Any]:
		"""
		Load checkpoint.

		Args:
		    checkpoint_name: Checkpoint filename or path
		    device: Device to load checkpoint to

		Returns:
		    Checkpoint dictionary

		Raises:
		    FileNotFoundError: If checkpoint doesn't exist
		"""
		if os.path.isabs(checkpoint_name):
			path = Path(checkpoint_name)
		else:
			path = self.checkpoint_dir / checkpoint_name

		if not path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {path}")

		checkpoint = torch.load(path, map_location=device)
		print(f"Loaded checkpoint: {path}")

		return checkpoint

	def get_latest_checkpoint(self) -> Optional[str]:
		"""
		Get path to latest checkpoint.

		Returns:
		    Path to latest.chkpt, or None if doesn't exist
		"""
		latest_path = self.checkpoint_dir / 'latest.chkpt'
		if latest_path.exists():
			return str(latest_path)
		return None

	def get_best_checkpoint(self) -> Optional[str]:
		"""
		Get path to best checkpoint.

		Returns:
		    Path to best checkpoint, or None if none exist
		"""
		pattern = f'best_ep*_{self.monitor_field}_*.chkpt'
		checkpoints = glob.glob(str(self.checkpoint_dir / pattern))

		if not checkpoints:
			return None

		# Return most recent (highest episode number)
		checkpoints.sort()
		return checkpoints[-1]
