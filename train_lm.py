#!/usr/bin/env python
"""
Training script for attention-based language models.

Usage:
    python train_lm.py trigo-gpt2                        # Use config name
    python train_lm.py configs/training/trigo-gpt2.yaml  # Use config path
    python train_lm.py trigo-gpt2 training.epochs=50     # With overrides
    python train_lm.py outputs/trigor/20251113-trigo-gpt2/  # Resume from experiment dir
    python train_lm.py outputs/trigor/20251113-trigo-gpt2/ training.epochs=100  # Resume with overrides
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Load environment variables from .env.local (for wandb API keys, etc.)
load_dotenv(dotenv_path='.env.local')

# Register custom OmegaConf resolver for date
OmegaConf.register_new_resolver("date", lambda: datetime.now().strftime("%Y%m%d"))

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trigor.data import TGNDataset, make_dataset
from trigor.training.lm_trainer import LMTrainer


# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
	"""
	Parse command-line arguments manually (without Hydra).

	Supports:
	  - Positional config: train_lm.py trigo-gpt2
	  - Config path: train_lm.py configs/training/trigo-gpt2.yaml
	  - Experiment directory (resume): train_lm.py outputs/trigor/20251113-trigo-gpt2
	  - Key=value overrides: train_lm.py trigo-gpt2 training.epochs=50

	Returns:
	    tuple: (config_path_or_resume_dir, overrides_dict, is_resume)
	"""
	if len(sys.argv) < 2 or sys.argv[1] in ['--help', '-h']:
		print("Usage: train_lm.py <config_name_or_path_or_experiment_dir> [key=value overrides...]")
		print("\nExamples:")
		print("  train_lm.py trigo-gpt2")
		print("  train_lm.py configs/training/trigo-gpt2.yaml")
		print("  train_lm.py outputs/trigor/20251113-trigo-gpt2  # Resume")
		print("  train_lm.py trigo-gpt2 training.epochs=50")
		print("\nAvailable configs:")
		for cfg in sorted(Path("configs/training").glob("*.yaml")):
			if not cfg.name.startswith("_"):
				print(f"  - {cfg.stem}")
		sys.exit(0)

	first_arg = sys.argv[1]
	arg_path = Path(first_arg)

	# Parse CLI overrides (key=value arguments)
	overrides = {}
	for arg in sys.argv[2:]:
		if '=' in arg:
			key, value = arg.split('=', 1)
			# Try to parse as number, boolean, or keep as string
			try:
				if value.lower() in ['true', 'false']:
					value = value.lower() == 'true'
				elif value.lower() == 'null' or value.lower() == 'none':
					value = None
				else:
					# Try int first, then float
					try:
						value = int(value)
					except ValueError:
						try:
							value = float(value)
						except ValueError:
							pass  # Keep as string
			except:
				pass  # Keep as string

			# Set nested key using dot notation
			keys = key.split('.')
			current = overrides
			for k in keys[:-1]:
				if k not in current:
					current[k] = {}
				current = current[k]
			current[keys[-1]] = value

	# Case 1: Experiment directory (resume training)
	if arg_path.is_dir():
		config_file = arg_path / "config.yaml"
		checkpoint_file = arg_path / "checkpoints" / "latest.chkpt"

		if config_file.exists() and checkpoint_file.exists():
			logger.info(f"Detected experiment directory: {arg_path}")
			logger.info(f"Will resume training from: {checkpoint_file}")
			return str(arg_path.resolve()), overrides, True
		else:
			logger.error(f"Invalid experiment directory: {arg_path}")
			if not config_file.exists():
				logger.error(f"  Missing config file: {config_file}")
			if not checkpoint_file.exists():
				logger.error(f"  Missing checkpoint: {checkpoint_file}")
			sys.exit(1)

	# Case 2: Full path to config file
	elif arg_path.suffix in ['.yaml', '.yml']:
		if not arg_path.exists():
			logger.error(f"Config file not found: {arg_path}")
			sys.exit(1)
		return str(arg_path), overrides, False

	# Case 3: Short config name (e.g., trigo-gpt2)
	else:
		config_path = Path("configs/training") / f"{first_arg}.yaml"
		if not config_path.exists():
			logger.error(f"Config file not found: {config_path}")
			logger.error(f"Available configs in configs/training/:")
			for cfg in Path("configs/training").glob("*.yaml"):
				if not cfg.name.startswith("_"):
					logger.error(f"  - {cfg.stem}")
			sys.exit(1)
		return str(config_path), overrides, False


def apply_overrides(config: DictConfig, overrides: dict) -> DictConfig:
	"""Apply override dictionary to config."""
	if not overrides:
		return config

	override_config = OmegaConf.create(overrides)
	return OmegaConf.merge(config, override_config)


def set_env_from_config(config: DictConfig):
	"""
	Set environment variables from config.

	Reads the top-level 'env' section and sets os.environ.
	This affects the entire program execution.

	Args:
	    config: Configuration object with optional 'env' section
	"""
	if not config.get('env'):
		return

	env_vars = OmegaConf.to_container(config.env, resolve=True)
	if not env_vars:
		return

	logger.info("")
	logger.info("Setting global environment variables from config:")
	for key, value in env_vars.items():
		str_value = str(value)
		os.environ[key] = str_value
		logger.info(f"  {key}: {str_value}")


def set_seed(seed: int, deterministic: bool = False):
	"""Set random seeds for reproducibility."""
	import random
	import numpy as np

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	if deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def create_dataloaders(config: DictConfig) -> tuple:
	"""
	Create train and validation dataloaders.

	Args:
	    config: Configuration object

	Returns:
	    Tuple of (train_loader, val_loader)
	"""
	logger.info("=" * 80)
	logger.info("Creating Datasets")
	logger.info("=" * 80)

	# Create training dataset
	train_config = OmegaConf.to_container(config.data, resolve=True)
	train_config['split'] = config.data.train_split
	train_dataset = make_dataset(config.data.type, train_config)

	# Create validation dataset
	val_dataset = None
	if config.data.get('val_split', None):
		val_config = OmegaConf.to_container(config.data, resolve=True)
		val_config['split'] = config.data.val_split
		val_dataset = make_dataset(config.data.type, val_config)

	# Create dataloaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.data.loader.batch_size,
		shuffle=config.data.loader.shuffle,
		num_workers=config.data.loader.num_workers,
		pin_memory=config.data.loader.pin_memory,
		collate_fn=TGNDataset.collate_batch,
		drop_last=True,  # Drop incomplete batches
	)

	val_loader = None
	if val_dataset:
		val_loader = DataLoader(
			val_dataset,
			batch_size=config.data.loader.batch_size,
			shuffle=False,  # Don't shuffle validation
			num_workers=config.data.loader.num_workers,
			pin_memory=config.data.loader.pin_memory,
			collate_fn=TGNDataset.collate_batch,
			drop_last=False,
		)

	logger.info("")
	logger.info("Datasets created:")
	logger.info(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
	if val_loader:
		logger.info(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

	return train_loader, val_loader


def main():
	"""Main training entry point."""
	# Parse command-line arguments
	config_path_or_dir, cli_overrides, is_resume = parse_arguments()

	if is_resume:
		# Resuming from experiment directory
		resume_path = Path(config_path_or_dir)
		saved_config_file = resume_path / "config.yaml"
		checkpoint_file = resume_path / "checkpoints" / "latest.chkpt"

		logger.info("=" * 80)
		logger.info("Resuming Training from Experiment Directory")
		logger.info("=" * 80)
		logger.info(f"Experiment directory: {resume_path}")
		logger.info(f"Loading config from: {saved_config_file}")
		logger.info(f"Loading checkpoint from: {checkpoint_file}")
		if cli_overrides:
			logger.info(f"CLI overrides: {cli_overrides}")
		logger.info("")

		# Load saved config (already resolved)
		config = OmegaConf.load(saved_config_file)

		# Apply CLI overrides if any (these take priority)
		config = apply_overrides(config, cli_overrides)

		# Use the same output directory as before
		output_dir = resume_path
		checkpoint_file = str(checkpoint_file)
	else:
		# Starting new training
		logger.info("=" * 80)
		logger.info("Starting New Training")
		logger.info("=" * 80)
		logger.info(f"Loading config from: {config_path_or_dir}")
		if cli_overrides:
			logger.info(f"CLI overrides: {cli_overrides}")
		logger.info("")

		# Load config file
		config = OmegaConf.load(config_path_or_dir)

		# Apply CLI overrides if any
		config = apply_overrides(config, cli_overrides)

		# Resolve interpolations (like ${date:}, ${paths.root})
		OmegaConf.resolve(config)

		# Setup output directory based on id
		output_dir = Path(config.paths.output) / config.id
		output_dir.mkdir(parents=True, exist_ok=True)
		checkpoint_file = None

	# Setup file logging
	log_file = output_dir / "train.log"
	# Use append mode if resuming, otherwise create new
	log_mode = 'a' if is_resume else 'w'
	file_handler = logging.FileHandler(log_file, mode=log_mode)
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(logging.Formatter(
		'%(asctime)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	))
	logging.getLogger().addHandler(file_handler)

	# Save/update config to output directory (with all variables resolved)
	config_file = output_dir / "config.yaml"
	# Resolve all variable interpolations before saving
	resolved_config = OmegaConf.to_container(config, resolve=True)
	resolved_config_obj = OmegaConf.create(resolved_config)
	with open(config_file, 'w') as f:
		f.write(OmegaConf.to_yaml(resolved_config_obj))

	if not is_resume:
		logger.info("=" * 80)
		logger.info("Attention Language Model Training")
		logger.info("=" * 80)
		logger.info(f"Experiment ID: {config.id}")
		logger.info(f"Output directory: {output_dir}")
		logger.info(f"Config saved to: {config_file}")
		logger.info(f"Log file: {log_file}")
	else:
		logger.info("")
		logger.info("Resuming training...")
		logger.info(f"Config updated at: {config_file}")
		logger.info(f"Log file (append mode): {log_file}")

	# Set global environment variables from config (affects entire program)
	set_env_from_config(config)

	# Print config
	logger.info("")
	logger.info("Configuration:")
	logger.info(OmegaConf.to_yaml(config))

	# Set random seeds
	logger.info("=" * 80)
	logger.info("Setting Random Seeds")
	logger.info("=" * 80)
	set_seed(config.seed, config.deterministic)
	logger.info(f"  Seed: {config.seed}")
	logger.info(f"  Deterministic: {config.deterministic}")

	# Check device
	if config.device == 'cuda' and not torch.cuda.is_available():
		logger.warning("CUDA requested but not available, using CPU")
		config.device = 'cpu'

	logger.info("")
	logger.info(f"Device: {config.device}")
	if config.device == 'cuda':
		logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
		logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

	# Create dataloaders
	train_loader, val_loader = create_dataloaders(config)

	# Create trainer (model will be created internally)
	logger.info("")
	logger.info("=" * 80)
	logger.info("Creating Trainer")
	logger.info("=" * 80)
	trainer = LMTrainer(
		config=config,
		train_loader=train_loader,
		val_loader=val_loader,
	)

	# Resume from checkpoint if specified
	if checkpoint_file:
		# Resume from experiment directory
		trainer.load_checkpoint(str(checkpoint_file))
	elif config.get('resume_from', None):
		# Resume from explicitly specified checkpoint path
		trainer.load_checkpoint(config.resume_from)

	# Start training
	trainer.train()

	logger.info("")
	logger.info("Training completed successfully!")


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		logger.error(f"Training failed with error: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
