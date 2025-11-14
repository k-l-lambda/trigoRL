"""
Language Model Trainer for attention-based models.

Provides epoch-based training loop with wandb logging, checkpointing,
and learning rate scheduling.
"""

import itertools
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from trigor.models import make_model
from trigor.utils.checkpoint import CheckpointManager
from trigor.utils.logger import WandbLogger


logger = logging.getLogger(__name__)


class LMTrainer:
	"""
	Trainer for language models with attention mechanisms.

	Supports:
	- Epoch-based training with gradient accumulation
	- Learning rate warmup and cosine annealing
	- Gradient clipping
	- Wandb logging
	- Checkpointing (best/latest)
	- Resume from checkpoint
	"""

	def __init__(
		self,
		config: DictConfig,
		train_loader: DataLoader,
		val_loader: Optional[DataLoader] = None,
	):
		"""
		Initialize LM trainer.

		Args:
		    config: OmegaConf configuration object
		    train_loader: Training data loader
		    val_loader: Optional validation data loader
		"""
		self.config = config
		self.train_loader = train_loader
		self.val_loader = val_loader

		# Set environment variables from config
		self._set_env_variables()

		# Create model from config
		logger.info("=" * 80)
		logger.info("Creating Model")
		logger.info("=" * 80)
		self.model = self._create_model()
		self.model = self.model.to(config.device)

		# Print model info
		num_params = sum(p.numel() for p in self.model.parameters())
		num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		logger.info("")
		logger.info("Model created:")
		logger.info(f"  Type: {config.model.type}")
		logger.info(f"  Base model: {config.model.config.model_config.type}")
		logger.info(f"  Total parameters: {num_params:,}")
		logger.info(f"  Trainable parameters: {num_trainable:,}")

		# Training state
		self.current_epoch = 0
		self.global_step = 0
		self.global_examples = 0  # Total examples processed (for logging)
		self.validation_count = 0  # Number of validations performed
		self.best_val_metric = float('inf') if config.training.monitor.mode == 'min' else float('-inf')

		# Setup optimizer
		self.optimizer = self._setup_optimizer()

		# Setup scheduler
		self.scheduler = self._setup_scheduler()

		# Setup wandb logger
		self.logger = None
		if config.training.wandb.enabled:
			# Use environment variables as defaults for null config values
			wandb_entity = os.getenv('WANDB_ENTITY')
			wandb_project = os.getenv('WANDB_PROJECT', 'trigor')
			# Use config.id as wandb run name for consistency
			wandb_name = config.training.wandb.get('name', config.id)

			self.logger = WandbLogger(
				project=wandb_project,
				entity=wandb_entity,
				name=wandb_name,
				config=OmegaConf.to_container(config, resolve=True),
				tags=config.training.wandb.tags,
				enabled=True,
			)
			# Watch model (gradients and parameters)
			self.logger.watch_model(self.model, log='all', log_freq=config.training.log_frequency)

		# Setup checkpoint manager
		checkpoint_dir = Path(config.paths.output) / config.id / "checkpoints"
		self.checkpoint_mgr = CheckpointManager(
			checkpoint_dir=checkpoint_dir,
			save_mode=config.training.save_mode,
			monitor_field=config.training.monitor.field,
			monitor_mode=config.training.monitor.mode,
			keep_n_checkpoints=config.training.keep_n_checkpoints,
		)

		logger.info("")
		logger.info("Trainer initialized:")
		logger.info(f"  Device: {config.device}")
		logger.info(f"  Checkpoint dir: {checkpoint_dir}")
		logger.info(f"  Training samples: {len(train_loader.dataset)}")
		logger.info(f"  Validation samples: {len(val_loader.dataset) if val_loader else 0}")
		logger.info(f"  Batch size: {config.data.loader.batch_size}")
		logger.info(f"  Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
		logger.info(f"  Effective batch size: {config.data.loader.batch_size * config.training.gradient_accumulation_steps}")

		# Show epoch size info
		epoch_size = config.training.get('epoch_size', None)
		if epoch_size is not None:
			batches_per_epoch = (epoch_size + config.data.loader.batch_size - 1) // config.data.loader.batch_size
			logger.info(f"  Epoch size: {epoch_size} examples (~{batches_per_epoch} batches)")
		else:
			logger.info(f"  Epoch size: full dataset ({len(train_loader)} batches)")

		logger.info(f"  Epochs: {config.training.epochs}")
		logger.info(f"  Learning rate: {config.training.learning_rate}")
		logger.info(f"  Warmup steps: {config.training.warmup_steps}")
		logger.info(f"  Log frequency: every {config.training.log_frequency} examples")
		logger.info(f"  Wandb logging: {'enabled' if config.training.wandb.enabled else 'disabled'}")


	def _set_env_variables(self):
		"""
		Set trainer-specific environment variables from config.

		Reads the 'training.env' section of config and sets os.environ accordingly.
		This only affects trainer operations, not the entire program.
		Useful for setting trainer-specific parameters.
		"""
		if not self.config.training.get('env'):
			return

		env_vars = OmegaConf.to_container(self.config.training.env, resolve=True)
		if not env_vars:
			return

		logger.info("")
		logger.info("Setting trainer environment variables from config:")
		for key, value in env_vars.items():
			# Convert value to string (in case it's a number)
			str_value = str(value)
			os.environ[key] = str_value
			logger.info(f"  {key}: {str_value}")


	def _create_model(self) -> nn.Module:
		"""
		Create model from config using factory pattern.

		Returns:
		    Model instance
		"""
		return make_model(self.config.model.type, self.config.model.config)


	def _setup_optimizer(self) -> AdamW:
		"""Create AdamW optimizer."""
		return AdamW(
			self.model.parameters(),
			lr=self.config.training.learning_rate,
			weight_decay=self.config.training.weight_decay,
			betas=(0.9, 0.999),
			eps=1e-8,
		)


	def _setup_scheduler(self) -> Optional[SequentialLR]:
		"""Create learning rate scheduler with warmup."""
		total_steps = len(self.train_loader) * self.config.training.epochs
		warmup_steps = self.config.training.warmup_steps
		scheduler_type = self.config.training.scheduler.type

		# Special case: LambdaLR with inverse square root (no separate warmup needed)
		if scheduler_type == 'inverse_sqrt':
			d_model = self.config.training.scheduler.get('d_model', 512)
			lr_mul = self.config.training.scheduler.get('lr_mul', 1.0)

			def lr_lambda(current_step):
				if current_step == 0:
					current_step = 1

				# Scale factor from model dimension
				scale = d_model ** -0.5

				# Warmup or inverse sqrt decay
				step_scale = min(current_step ** (-0.5),
				                current_step * warmup_steps ** (-1.5))

				return lr_mul * scale * step_scale

			return LambdaLR(self.optimizer, lr_lambda)

		# Special case: LambdaLR with custom lambda function
		if scheduler_type == 'lambda':
			# User provides a lambda string that will be evaluated
			# For safety, only allow if explicitly configured
			lambda_str = self.config.training.scheduler.get('lambda_fn', None)
			if lambda_str:
				logger.warning("Using custom lambda function for learning rate scheduling")
				logger.warning(f"Lambda: {lambda_str}")
				# Create lambda function from string (be careful with this!)
				lr_lambda = eval(lambda_str)
				return LambdaLR(self.optimizer, lr_lambda)
			else:
				raise ValueError("scheduler.type='lambda' requires 'lambda_fn' to be specified")

		# Standard schedulers with optional warmup
		if warmup_steps == 0 and scheduler_type == 'constant':
			# No scheduler needed
			return None

		schedulers = []
		milestones = []

		# Warmup phase
		if warmup_steps > 0:
			warmup_scheduler = LinearLR(
				self.optimizer,
				start_factor=0.01,
				end_factor=1.0,
				total_iters=warmup_steps,
			)
			schedulers.append(warmup_scheduler)
			milestones.append(warmup_steps)

		# Main scheduler phase
		main_steps = total_steps - warmup_steps
		if scheduler_type == 'cosine':
			main_scheduler = CosineAnnealingLR(
				self.optimizer,
				T_max=main_steps,
				eta_min=self.config.training.scheduler.min_lr,
			)
			schedulers.append(main_scheduler)
		elif scheduler_type == 'linear':
			main_scheduler = LinearLR(
				self.optimizer,
				start_factor=1.0,
				end_factor=self.config.training.scheduler.min_lr / self.config.training.learning_rate,
				total_iters=main_steps,
			)
			schedulers.append(main_scheduler)
		# 'constant' type means no decay after warmup

		if len(schedulers) == 0:
			return None
		elif len(schedulers) == 1:
			return schedulers[0]
		else:
			return SequentialLR(
				self.optimizer,
				schedulers=schedulers,
				milestones=milestones,
			)


	def train(self):
		"""Main training loop."""
		logger.info("")
		logger.info("=" * 80)
		logger.info("Starting Training")
		logger.info("=" * 80)

		# Evaluate with initial weights (only if starting from scratch)
		if self.current_epoch == 0 and self.val_loader:
			logger.info("")
			logger.info("Evaluating initial model (epoch 0)...")
			initial_metrics = self._validate_epoch()
			logger.info("")
			logger.info("Initial model metrics:")
			logger.info(f"  Val Loss: {initial_metrics['val_loss']:.4f}")
			logger.info(f"  Val Error: {initial_metrics['val_error']:.4f}")
			logger.info(f"  Val Perplexity: {initial_metrics['val_perplexity']:.2f}")
			logger.info("")

		try:
			for epoch in range(self.current_epoch, self.config.training.epochs):
				self.current_epoch = epoch

				# Training phase
				train_metrics = self._train_epoch()

				# Validation phase
				val_metrics = {}
				if self.val_loader and (epoch % self.config.eval.eval_frequency == 0):
					val_metrics = self._validate_epoch()
					self.validation_count += 1

					# Save checkpoint based on validation count
					if self.validation_count % self.config.training.save_frequency == 0:
						self._save_checkpoint(val_metrics)

				# Log epoch summary
				self._log_epoch_summary(epoch, train_metrics, val_metrics)

		except KeyboardInterrupt:
			logger.warning("")
			logger.warning("Training interrupted by user")
			logger.info("Saving checkpoint before exit...")
			self._save_checkpoint({})

		finally:
			if self.logger:
				self.logger.finish()

		logger.info("")
		logger.info("=" * 80)
		logger.info("Training Complete")
		logger.info("=" * 80)


	def _train_epoch(self) -> Dict[str, float]:
		"""Train for one epoch."""
		self.model.train()

		# Accumulators
		total_loss = 0.0
		total_error = 0.0
		total_perplexity = 0.0
		total_top5_error = 0.0
		total_tokens = 0
		num_batches = 0

		# Determine number of batches for this epoch
		epoch_size = self.config.training.get('epoch_size', None)
		if epoch_size is not None:
			# Calculate batches needed for epoch_size examples
			batch_size = self.config.data.loader.batch_size
			max_batches = (epoch_size + batch_size - 1) // batch_size  # Ceiling division
			# Create infinite iterator from train_loader
			data_iterator = itertools.cycle(self.train_loader)
		else:
			# Use full dataset (one pass)
			max_batches = len(self.train_loader)
			data_iterator = iter(self.train_loader)

		# Progress bar
		pbar = tqdm(
			itertools.islice(data_iterator, max_batches),
			total=max_batches,
			desc=f"Epoch {self.current_epoch+1}/{self.config.training.epochs} [Train]"
		)

		self.optimizer.zero_grad()

		for batch_idx, batch in enumerate(pbar):
			# Move batch to device
			input_ids = batch['input_ids'].to(self.config.device)
			labels = batch['labels'].to(self.config.device)
			attention_mask = batch['attention_mask'].to(self.config.device)

			# Forward pass
			outputs = self.model(input_ids, labels, attention_mask)

			# Scale loss for gradient accumulation
			loss = outputs['loss'] / self.config.training.gradient_accumulation_steps
			loss.backward()

			# Accumulate metrics (unscaled)
			total_loss += outputs['loss'].item()
			total_error += outputs['error'].item()
			total_perplexity += outputs['perplexity'].item()
			total_top5_error += outputs['top5_error'].item()
			total_tokens += outputs['num_tokens'].item()
			num_batches += 1

			# Update progress bar
			pbar.set_postfix({
				'loss': f"{outputs['loss'].item():.4f}",
				'acc': f"{outputs['error'].item():.4f}",
				'ppl': f"{outputs['perplexity'].item():.2f}",
				'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
			})

			# Optimizer step (after accumulation)
			if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
				# Gradient clipping
				if self.config.training.max_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(
						self.model.parameters(),
						self.config.training.max_grad_norm,
					)

				# Optimizer step
				self.optimizer.step()

				# Scheduler step
				if self.scheduler:
					self.scheduler.step()

				# Zero gradients
				self.optimizer.zero_grad()

				# Increment global step and examples
				self.global_step += 1
				current_batch_size = input_ids.size(0)
				self.global_examples += current_batch_size

				# Log to wandb (based on examples processed)
				if self.logger and (self.global_examples % self.config.training.log_frequency == 0):
					self.logger.log({
						'train/loss': outputs['loss'].item(),
						'train/error': outputs['error'].item(),
						'train/perplexity': outputs['perplexity'].item(),
						'train/top5_error': outputs['top5_error'].item(),
						'train/learning_rate': self.optimizer.param_groups[0]['lr'],
					}, step=self.global_examples)

		pbar.close()

		# Compute averages
		avg_metrics = {
			'loss': total_loss / num_batches,
			'error': total_error / num_batches,
			'perplexity': total_perplexity / num_batches,
			'top5_error': total_top5_error / num_batches,
			'tokens': total_tokens,
		}

		return avg_metrics


	def _validate_epoch(self) -> Dict[str, float]:
		"""Validate for one epoch."""
		self.model.eval()

		# Accumulators
		total_loss = 0.0
		total_error = 0.0
		total_perplexity = 0.0
		total_top5_error = 0.0
		total_tokens = 0
		num_batches = 0

		# Limit validation batches if specified
		max_batches = self.config.eval.get('eval_batches', None)

		# Progress bar
		pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.training.epochs} [Val]")

		with torch.no_grad():
			for batch_idx, batch in enumerate(pbar):
				# Stop if max_batches reached
				if max_batches and batch_idx >= max_batches:
					break

				# Move batch to device
				input_ids = batch['input_ids'].to(self.config.device)
				labels = batch['labels'].to(self.config.device)
				attention_mask = batch['attention_mask'].to(self.config.device)

				# Forward pass
				outputs = self.model(input_ids, labels, attention_mask)

				# Accumulate metrics
				total_loss += outputs['loss'].item()
				total_error += outputs['error'].item()
				total_perplexity += outputs['perplexity'].item()
				total_top5_error += outputs['top5_error'].item()
				total_tokens += outputs['num_tokens'].item()
				num_batches += 1

				# Update progress bar
				pbar.set_postfix({
					'loss': f"{outputs['loss'].item():.4f}",
					'err': f"{outputs['error'].item():.4f}",
					'acc': f"{(1 - outputs['error'].item()):.4f}",
					'ppl': f"{outputs['perplexity'].item():.2f}",
				})

		pbar.close()

		# Compute averages
		avg_metrics = {
			'val_loss': total_loss / num_batches,
			'val_error': total_error / num_batches,
			'val_perplexity': total_perplexity / num_batches,
			'val_top5_error': total_top5_error / num_batches,
			'val_tokens': total_tokens,
		}

		# Log to wandb
		if self.logger:
			self.logger.log(avg_metrics, step=self.global_examples)

		return avg_metrics


	def _log_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
		"""Log epoch summary."""
		logger.info("")
		logger.info(f"Epoch {epoch+1}/{self.config.training.epochs} Summary:")
		logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
		logger.info(f"  Train Error: {train_metrics['error']:.4f}")
		logger.info(f"  Train Perplexity: {train_metrics['perplexity']:.2f}")

		if val_metrics:
			logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
			logger.info(f"  Val Error: {val_metrics['val_error']:.4f}")
			logger.info(f"  Val Perplexity: {val_metrics['val_perplexity']:.2f}")


	def _save_checkpoint(self, val_metrics: Dict):
		"""Save checkpoint."""
		checkpoint = {
			'epoch': self.current_epoch,
			'global_step': self.global_step,
			'global_examples': self.global_examples,  # Save examples count for logging continuity
			'validation_count': self.validation_count,  # Save validation count for save_frequency
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
			'best_val_metric': self.best_val_metric,
			'config': OmegaConf.to_container(self.config, resolve=True),
		}

		# Get metric value for checkpoint manager
		monitor_field = self.config.training.monitor.field
		metric_value = val_metrics.get(monitor_field, None)

		# Save checkpoint
		checkpoint_path = self.checkpoint_mgr.save(
			checkpoint=checkpoint,
			episode=self.current_epoch,  # Use epoch as episode for compatibility
			metric_value=metric_value,
			is_latest=True,
		)

		if checkpoint_path:
			logger.info(f"Checkpoint saved: {checkpoint_path}")

			# Upload to wandb
			if self.logger:
				self.logger.save_checkpoint(checkpoint_path)

			# Update best metric
			if metric_value is not None and self.checkpoint_mgr.is_new_best(metric_value):
				self.best_val_metric = metric_value
				logger.info(f"New best {monitor_field}: {metric_value:.4f}")


	def load_checkpoint(self, checkpoint_path: Optional[str] = None):
		"""
		Load checkpoint and resume training.

		Args:
		    checkpoint_path: Path to checkpoint file. If None, loads latest.
		"""
		if checkpoint_path is None:
			checkpoint_path = self.checkpoint_mgr.get_latest_checkpoint()

		if checkpoint_path is None:
			logger.warning("No checkpoint found to load")
			return

		logger.info(f"Loading checkpoint: {checkpoint_path}")
		checkpoint = self.checkpoint_mgr.load(checkpoint_path, device=self.config.device)

		# Restore state
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		if self.scheduler and checkpoint['scheduler_state_dict']:
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

		self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
		self.global_step = checkpoint['global_step']
		self.global_examples = checkpoint.get('global_examples', 0)  # Restore examples count (default 0 for old checkpoints)
		self.validation_count = checkpoint.get('validation_count', 0)  # Restore validation count (default 0 for old checkpoints)
		self.best_val_metric = checkpoint['best_val_metric']

		logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}, examples {self.global_examples}, validations {self.validation_count}")
