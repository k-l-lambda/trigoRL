#!/usr/bin/env python
"""
Training script for attention-based language models.

Usage:
    python train_lm.py                                   # Use default config (trigo-gpt2)
    python train_lm.py trigo-llama                      # Use specific config (short name)
    python train_lm.py configs/training/trigo-rwkv.yaml # Use config file path
    python train_lm.py trigo-gpt2 training.epochs=50    # Config + overrides
    python train_lm.py --config-name=trigo-rwkv          # Alternative syntax
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
	Parse positional argument as config name/path.

	Supports:
	  - Short name: trigo-gpt2
	  - Relative path: configs/training/trigo-gpt2.yaml
	  - Absolute path: /path/to/config.yaml

	Converts to Hydra's --config-name format.
	"""
	# Check if first argument is a positional config (not a Hydra override)
	if len(sys.argv) > 1:
		first_arg = sys.argv[1]

		# Skip if it's already a Hydra parameter
		if first_arg.startswith('-') or '=' in first_arg:
			return

		# Parse the positional argument
		config_path = Path(first_arg)

		# Case 1: Full path to config file
		if config_path.suffix in ['.yaml', '.yml']:
			config_name = config_path.stem  # Get name without extension

			# If it's a relative path starting with configs/training/
			if str(config_path).startswith('configs/training/'):
				# Just use the config name
				sys.argv[1] = f'--config-name={config_name}'
			else:
				# For other paths, we need to handle config_path too
				# For simplicity, just use the name and assume default path
				sys.argv[1] = f'--config-name={config_name}'

		# Case 2: Short name (e.g., trigo-gpt2)
		else:
			config_name = first_arg
			sys.argv[1] = f'--config-name={config_name}'


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
	logger.info("=" * 80)
	logger.info("Attention Language Model Training")
	logger.info("=" * 80)

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
	if config.get('resume_from', None):
		trainer.load_checkpoint(config.resume_from)

	# Start training
	trainer.train()

	logger.info("")
	logger.info("Training completed successfully!")


if __name__ == "__main__":
	# Parse positional config argument before Hydra processes sys.argv
	parse_positional_config()

	try:
		main()
	except Exception as e:
		logger.error(f"Training failed with error: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
