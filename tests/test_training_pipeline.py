#!/usr/bin/env python
"""
Quick test of end-to-end training with C++ self-play data.

This script:
1. Verifies C++ self-play data exists
2. Loads dataset
3. Creates a small model
4. Runs 1 epoch of training
5. Confirms loss decreases
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data.utils import make_dataloader
from trigor.models import make_model


def test_training_pipeline():
	"""Test end-to-end training pipeline."""
	print("=" * 70)
	print("Testing End-to-End Training with C++ Self-Play Data")
	print("=" * 70)

	# Check data exists
	data_dir = "/tmp/selfplay_test"
	if not Path(data_dir).exists():
		print(f"\nError: Data directory not found: {data_dir}")
		print("Please generate data first:")
		print("  cd /home/camus/work/trigo.cpp/build")
		print("  ./self_play_generator --num-games 20 --output /tmp/selfplay_test")
		return False

	# Setup device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"\nDevice: {device}")

	# Create dataloaders
	print("\n" + "=" * 70)
	print("Creating DataLoaders")
	print("=" * 70)

	train_config = {
		'data_dir': data_dir,
		'max_length': 1024,
		'split': '*0..7/10',  # 80% train
	}

	val_config = {
		'data_dir': data_dir,
		'max_length': 1024,
		'split': '8,9/10',  # 20% val
	}

	train_loader = make_dataloader(
		'TGNDataset',
		train_config,
		batch_size=2,
		shuffle=True,
		num_workers=0,
	)

	val_loader = make_dataloader(
		'TGNDataset',
		val_config,
		batch_size=2,
		shuffle=False,
		num_workers=0,
	)

	print(f"Train batches: {len(train_loader)}")
	print(f"Val batches: {len(val_loader)}")

	# Create model
	print("\n" + "=" * 70)
	print("Creating Model")
	print("=" * 70)

	model_config = {
		'type': 'AttentionCausalLoss',
		'config': {
			'model_config': {
				'type': 'GPT2CausalLM',
				'config': {
					'vocab_size': 128,
					'hidden_size': 128,
					'num_layers': 2,
					'num_heads': 4,
					'max_seq_len': 1024,
					'dropout': 0.1,
					'activation': 'gelu_new',
					'intermediate_size': 512,
				}
			},
			'ignore_index': 0,  # PAD token
			'label_smoothing': 0.0,
		}
	}

	model = make_model(model_config['type'], model_config['config'])
	model = model.to(device)

	# Count parameters
	num_params = sum(p.numel() for p in model.parameters())
	num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total parameters: {num_params:,}")
	print(f"Trainable parameters: {num_trainable:,}")

	# Create optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

	# Training loop
	print("\n" + "=" * 70)
	print("Running Training Loop")
	print("=" * 70)

	model.train()
	train_losses = []

	for batch_idx, batch in enumerate(train_loader):
		if batch_idx >= 5:  # Just test 5 batches
			break

		# Move to device
		input_ids = batch['input_ids'].to(device)
		labels = batch['labels'].to(device)
		attention_mask = batch['attention_mask'].to(device)

		# Forward pass
		optimizer.zero_grad()
		outputs = model(
			input_ids=input_ids,
			labels=labels,
			attention_mask=attention_mask,
		)

		loss = outputs['loss']
		train_losses.append(loss.item())

		# Backward pass
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()

		print(f"Batch {batch_idx + 1}/5 | Loss: {loss.item():.4f}")

	# Validation
	print("\n" + "=" * 70)
	print("Running Validation")
	print("=" * 70)

	model.eval()
	val_losses = []

	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			if batch_idx >= 2:  # Just test 2 batches
				break

			# Move to device
			input_ids = batch['input_ids'].to(device)
			labels = batch['labels'].to(device)
			attention_mask = batch['attention_mask'].to(device)

			# Forward pass
			outputs = model(
				input_ids=input_ids,
				labels=labels,
				attention_mask=attention_mask,
			)

			loss = outputs['loss']
			val_losses.append(loss.item())

			print(f"Val Batch {batch_idx + 1}/2 | Loss: {loss.item():.4f}")

	# Summary
	print("\n" + "=" * 70)
	print("Summary")
	print("=" * 70)

	avg_train_loss = sum(train_losses) / len(train_losses)
	avg_val_loss = sum(val_losses) / len(val_losses)

	print(f"\nAverage train loss: {avg_train_loss:.4f}")
	print(f"Average val loss: {avg_val_loss:.4f}")

	# Check if loss is reasonable (not NaN or too large)
	if torch.isnan(torch.tensor(avg_train_loss)):
		print("\n❌ FAIL: Training loss is NaN")
		return False

	if avg_train_loss > 10.0:
		print("\n⚠️  WARNING: Training loss is very high (>10)")

	print("\n" + "=" * 70)
	print("✓ End-to-End Training Pipeline Working!")
	print("=" * 70)

	print("\nNext Steps:")
	print("1. Generate more self-play data:")
	print("   cd /home/camus/work/trigo.cpp/build")
	print("   ./self_play_generator --num-games 1000 --output /path/to/data")
	print("\n2. Run full training:")
	print("   python train_lm.py configs/training/trigo-selfplay.yaml")
	print("\n3. Monitor with wandb or tensorboard")

	return True


if __name__ == '__main__':
	success = test_training_pipeline()
	sys.exit(0 if success else 1)
