#!/usr/bin/env python
"""
Test script for batch visualization (non-interactive mode).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.view_dataset import load_dataset_from_config, visualize_batch
from torch.utils.data import DataLoader
from trigor.data import TGNDataset


def test_visualization():
	"""Test batch visualization in non-interactive mode."""
	config_path = project_root / "configs/training/trigo-gpt2.yaml"

	print(f"Loading dataset from: {config_path}")
	dataset = load_dataset_from_config(config_path)

	print(f"Dataset loaded: {len(dataset)} samples")

	# Create DataLoader
	batch_size = 4
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		collate_fn=TGNDataset.collate_batch,
	)

	print(f"Testing visualization with batch size {batch_size}...")

	# Get first batch
	batch = next(iter(dataloader))

	print(f"Batch shapes:")
	print(f"  input_ids: {batch['input_ids'].shape}")
	print(f"  labels: {batch['labels'].shape}")
	print(f"  attention_mask: {batch['attention_mask'].shape}")

	# Create visualization
	print("\nGenerating visualization...")

	# Temporarily override plt.show to save instead
	original_show = plt.show
	def save_instead(**kwargs):
		output_path = project_root / "test_batch_visualization.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		print(f"\n✓ Visualization saved to: {output_path}")
		plt.close('all')

	plt.show = save_instead

	try:
		visualize_batch(dataset, batch, 0, batch_size, len(dataloader))
		print("✓ Visualization test passed!")
	finally:
		plt.show = original_show


if __name__ == "__main__":
	test_visualization()
