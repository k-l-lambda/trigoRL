#!/usr/bin/env python
"""
Training script for attention-based language models.

Usage:
    python train_lm.py                                   # Use default config (trigo-gpt2)
    python train_lm.py trigo-llama                      # Use specific config (short name)
    python train_lm.py configs/training/trigo-rwkv.yaml # Use config file path
    python train_lm.py trigo-gpt2 training.epochs=50    # Config + overrides
    python train_lm.py --config-name=trigo-rwkv          # Alternative syntax
    python train_lm.py outputs/trigor/20251113-trigo-gpt2/  # Resume from experiment dir
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
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


def parse_positional_config():
	"""
	Parse positional argument as config name/path or experiment directory.

	Supports:
	  - Short name: trigo-gpt2
	  - Relative path: configs/training/trigo-gpt2.yaml
	  - Absolute path: /path/to/config.yaml
	  - Experiment directory: outputs/trigor/20251113-trigo-gpt2/

	Converts to Hydra's --config-name format.
	Returns experiment directory path if resuming, None otherwise.
	"""
	# Check if first argument is a positional config (not a Hydra override)
	if len(sys.argv) > 1:
		first_arg = sys.argv[1]

		# Skip if it's already a Hydra parameter
		if first_arg.startswith('-') or '=' in first_arg:
			return None

		# Parse the positional argument
		arg_path = Path(first_arg)

		# Case 1: Experiment directory (resume training)
		# Check if it's a directory with config.yaml and checkpoints/latest.chkpt
		if arg_path.is_dir():
			config_file = arg_path / "config.yaml"
			checkpoint_file = arg_path / "checkpoints" / "latest.chkpt"

			if config_file.exists() and checkpoint_file.exists():
				# This is a valid experiment directory - resume training
				logger.info(f"Detected experiment directory: {arg_path}")
				logger.info(f"Resuming training from: {checkpoint_file}")
				# Remove the directory argument and let Hydra use the saved config
				sys.argv.pop(1)
				# Store the experiment directory for later use
				return str(arg_path.resolve())
			else:
				logger.error(f"Invalid experiment directory: {arg_path}")
				if not config_file.exists():
					logger.error(f"  Missing config file: {config_file}")
				if not checkpoint_file.exists():
					logger.error(f"  Missing checkpoint: {checkpoint_file}")
				sys.exit(1)

		# Case 2: Full path to config file
		elif arg_path.suffix in ['.yaml', '.yml']:
			config_name = arg_path.stem  # Get name without extension

			# If it's a relative path starting with configs/training/
			if str(arg_path).startswith('configs/training/'):
				# Just use the config name
				sys.argv[1] = f'--config-name={config_name}'
			else:
				# For other paths, we need to handle config_path too
				# For simplicity, just use the name and assume default path
				sys.argv[1] = f'--config-name={config_name}'

		# Case 3: Short name (e.g., trigo-gpt2)
		else:
			config_name = first_arg
			sys.argv[1] = f'--config-name={config_name}'

	return None


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


@hydra.main(config_path="configs/training", config_name="trigo-gpt2", version_base=None)
def main(config: DictConfig):
	"""Main training entry point."""
	# Check if we're resuming from an experiment directory
	resume_dir = getattr(sys.modules['__main__'], '_resume_dir', None)

	# If resuming from experiment directory, load config from there
	if resume_dir:
		resume_path = Path(resume_dir)
		saved_config_file = resume_path / "config.yaml"
		checkpoint_file = resume_path / "checkpoints" / "latest.chkpt"

		logger.info("=" * 80)
		logger.info("Resuming Training from Experiment Directory")
		logger.info("=" * 80)
		logger.info(f"Experiment directory: {resume_path}")
		logger.info(f"Loading config from: {saved_config_file}")
		logger.info(f"Loading checkpoint from: {checkpoint_file}")
		logger.info("")

		# Load saved config
		saved_config = OmegaConf.load(saved_config_file)

		# Merge with any command-line overrides (config from Hydra may have overrides)
		# Priority: CLI overrides > saved config
		config = OmegaConf.merge(saved_config, config)

		# Use the same output directory as before
		output_dir = resume_path
	else:
		# Setup output directory based on id
		output_dir = Path(config.paths.output) / config.id
		output_dir.mkdir(parents=True, exist_ok=True)
		checkpoint_file = None

	# Setup file logging
	log_file = output_dir / "train.log"
	# Use append mode if resuming, otherwise create new
	log_mode = 'a' if resume_dir else 'w'
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

	if not resume_dir:
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
	# Parse positional config argument before Hydra processes sys.argv
	# Returns experiment directory if resuming, None otherwise
	resume_dir = parse_positional_config()

	try:
		# Pass resume_dir to main via a wrapper
		if resume_dir:
			# Hydra decorator doesn't support extra parameters directly
			# We need to store it and access it in main
			import __main__
			__main__._resume_dir = resume_dir

		main()
	except Exception as e:
		logger.error(f"Training failed with error: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
