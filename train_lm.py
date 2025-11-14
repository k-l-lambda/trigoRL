#!/usr/bin/env python
"""
Training script for attention-based language models.

Usage:
    python train_lm.py configs/training/trigo-gpt2.yaml  # Use config path
    python train_lm.py configs/training/trigo-gpt2.yaml training.epochs=50  # With overrides
    python train_lm.py outputs/trigor/20251113-trigo-gpt2/  # Resume from experiment dir
    python train_lm.py outputs/trigor/20251113-trigo-gpt2/ training.epochs=100  # Resume with overrides
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


# Global variable to store resume directory
_resume_dir = None


def preprocess_args():
	"""
	Preprocess command-line arguments before Hydra processes them.
	Converts positional config path/resume_dir into Hydra's expected format.

	Modifies sys.argv in place to make it compatible with Hydra.
	"""
	global _resume_dir

	if len(sys.argv) < 2 or sys.argv[1] in ['--help', '-h']:
		print("Usage: train_lm.py <config_path_or_experiment_dir> [key=value overrides...]")
		print("\nExamples:")
		print("  train_lm.py configs/training/trigo-gpt2.yaml")
		print("  train_lm.py outputs/trigor/20251113-trigo-gpt2  # Resume")
		print("  train_lm.py configs/training/trigo-gpt2.yaml training.epochs=50")
		print("\nAvailable configs:")
		for cfg in sorted(Path("configs/training").glob("*.yaml")):
			if not cfg.name.startswith("_"):
				print(f"  - {cfg.name}")
		sys.exit(0)

	first_arg = sys.argv[1]
	arg_path = Path(first_arg)

	# Case 1: Experiment directory (resume training)
	if arg_path.is_dir():
		config_file = arg_path / "config.yaml"
		checkpoint_file = arg_path / "checkpoints" / "latest.chkpt"

		if config_file.exists():
			if checkpoint_file.exists():
				# Resume training from checkpoint
				logger.info(f"Detected experiment directory: {arg_path}")
				logger.info(f"Will resume training from: {checkpoint_file}")
				_resume_dir = str(arg_path.resolve())
				# Use a default config, actual config will be loaded from saved file
				sys.argv[1:2] = ["--config-path=configs/training", "--config-name=trigo-gpt2"]
			else:
				# Config exists but no checkpoint - start training from scratch
				logger.info(f"Detected experiment directory without checkpoint: {arg_path}")
				logger.info("Will start training from scratch")
				# Load config from experiment directory
				config_name = "config"
				config_dir = str(arg_path.resolve())
				sys.argv[1:2] = [f"--config-path={config_dir}", f"--config-name={config_name}"]
			return
		else:
			logger.error(f"Invalid experiment directory: {arg_path}")
			logger.error(f"  Missing config file: {config_file}")
			sys.exit(1)

	# Case 2: Config file path
	elif arg_path.suffix in ['.yaml', '.yml']:
		if not arg_path.exists():
			logger.error(f"Config file not found: {arg_path}")
			sys.exit(1)
		# Extract config name and directory from path
		config_name = arg_path.stem
		config_dir = str(arg_path.parent.resolve())
		sys.argv[1:2] = [f"--config-path={config_dir}", f"--config-name={config_name}"]

	else:
		logger.error(f"Invalid argument: {first_arg}")
		logger.error("Please provide a config file path (.yaml) or experiment directory")
		sys.exit(1)


# Preprocess arguments before Hydra's decorator runs
preprocess_args()


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


@hydra.main(version_base=None)
def main(config: DictConfig):
	"""Main training entry point."""
	# Check if we're resuming from an experiment directory
	is_resume = _resume_dir is not None
	resume_wandb_id = None

	if is_resume:
		# Resuming from experiment directory
		resume_path = Path(_resume_dir)
		saved_config_file = resume_path / "config.yaml"
		checkpoint_file = resume_path / "checkpoints" / "latest.chkpt"

		logger.info("=" * 80)
		logger.info("Resuming Training from Experiment Directory")
		logger.info("=" * 80)
		logger.info(f"Experiment directory: {resume_path}")
		logger.info(f"Loading config from: {saved_config_file}")
		logger.info(f"Loading checkpoint from: {checkpoint_file}")
		logger.info("")

		# Load checkpoint to get wandb run_id
		if checkpoint_file.exists():
			checkpoint = torch.load(checkpoint_file, map_location='cpu')
			resume_wandb_id = checkpoint.get('wandb_run_id', None)
			if resume_wandb_id:
				logger.info(f"Found wandb run ID in checkpoint: {resume_wandb_id}")
				logger.info("Will resume logging to existing wandb run")
			del checkpoint  # Free memory

		# Load saved config (already resolved)
		saved_config = OmegaConf.load(saved_config_file)

		# Merge with any command-line overrides (config from Hydra may have overrides)
		# Priority: CLI overrides > saved config
		config = OmegaConf.merge(saved_config, config)

		# Use the same output directory as before
		output_dir = resume_path
		checkpoint_file = str(checkpoint_file)
	else:
		# Starting new training
		logger.info("=" * 80)
		logger.info("Starting New Training")
		logger.info("=" * 80)
		logger.info("")

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

	# Convert relative paths to absolute paths
	if 'paths' in resolved_config:
		for key, value in resolved_config['paths'].items():
			if value and isinstance(value, str):
				resolved_config['paths'][key] = str(Path(value).resolve())

	if 'data' in resolved_config and 'data_dir' in resolved_config['data']:
		data_dir = resolved_config['data']['data_dir']
		if data_dir and isinstance(data_dir, str):
			resolved_config['data']['data_dir'] = str(Path(data_dir).resolve())

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
		resume_wandb_id=resume_wandb_id,  # Pass wandb run ID for resume
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
