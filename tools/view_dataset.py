#!/usr/bin/env python
"""
CLI tool to view and validate TGNDataset contents.

This tool allows you to:
- Load a TGNDataset from a training config file
- View dataset statistics
- Display sample data with tokenization details
- Validate the dataset implementation
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trigor.data import TGNDataset


def load_dataset_from_config(config_path: Path) -> TGNDataset:
	"""
	Load TGNDataset from a training configuration file.

	Args:
	    config_path: Path to the config YAML file

	Returns:
	    Initialized TGNDataset
	"""
	# Load config
	cfg = OmegaConf.load(config_path)

	# Resolve paths relative to project root
	OmegaConf.update(cfg, "paths.root", str(project_root))
	OmegaConf.resolve(cfg)

	# Create dataset
	dataset = TGNDataset.from_config(cfg.data)

	return dataset


def display_dataset_stats(dataset: TGNDataset):
	"""Display comprehensive dataset statistics."""
	print("\n" + "=" * 80)
	print("DATASET STATISTICS")
	print("=" * 80)

	stats = dataset.get_stats()

	print(f"\nDataset: {dataset}")
	print(f"\nFiles:")
	print(f"  Total files:      {stats['num_files']}")
	print(f"  Total bytes:      {stats['total_bytes']:,} bytes ({stats['total_bytes'] / 1024:.1f} KB)")
	print(f"  Average size:     {stats['avg_bytes']:.1f} bytes")
	print(f"  Min size:         {stats['min_bytes']} bytes")
	print(f"  Max size:         {stats['max_bytes']} bytes")

	print(f"\nTokenization:")
	print(f"  Vocab size:       {stats['vocab_size']}")
	print(f"  Max sequence len: {stats['max_length']}")

	print(f"\nData Directory:")
	print(f"  {dataset.data_dir}")
	print()


def display_sample(
	dataset: TGNDataset,
	idx: int,
	show_tokens: bool = True,
	show_text: bool = True,
	show_decoded: bool = False,
	max_tokens_display: int = 50,
):
	"""
	Display a single sample from the dataset.

	Args:
	    dataset: TGNDataset instance
	    idx: Sample index
	    show_tokens: Show tokenized sequences
	    show_text: Show original text
	    show_decoded: Show decoded text from tokens
	    max_tokens_display: Maximum number of tokens to display
	"""
	print("=" * 80)
	print(f"SAMPLE {idx}")
	print("=" * 80)

	# Get file info
	file_info = dataset.get_file_info(idx)
	print(f"\nFile Information:")
	print(f"  Name:   {file_info['name']}")
	print(f"  Path:   {file_info['path']}")
	print(f"  Size:   {file_info['size_bytes']} bytes")

	# Get the sample
	sample = dataset[idx]

	# Display tensor shapes
	print(f"\nTensor Shapes:")
	print(f"  input_ids:      {sample['input_ids'].shape}")
	print(f"  labels:         {sample['labels'].shape}")
	print(f"  attention_mask: {sample['attention_mask'].shape}")

	# Count non-padding tokens
	non_pad_tokens = sample['attention_mask'].sum().item()
	print(f"\nToken Statistics:")
	print(f"  Non-padding tokens: {non_pad_tokens}")
	print(f"  Padding tokens:     {len(sample['input_ids']) - non_pad_tokens}")
	print(f"  Sequence length:    {len(sample['input_ids'])}")

	# Show original text
	if show_text:
		print(f"\nOriginal Text:")
		print("-" * 80)
		text = dataset.get_text(idx)
		# Show first 500 chars
		display_text = text[:500]
		if len(text) > 500:
			display_text += "\n... (truncated)"
		print(display_text)
		print("-" * 80)

	# Show tokens
	if show_tokens:
		print(f"\nInput Token IDs (first {max_tokens_display}):")
		input_tokens = sample['input_ids'][:max_tokens_display].tolist()
		print(f"  {input_tokens}")

		print(f"\nLabel Token IDs (first {max_tokens_display}):")
		label_tokens = sample['labels'][:max_tokens_display].tolist()
		print(f"  {label_tokens}")

		print(f"\nAttention Mask (first {max_tokens_display}):")
		mask = sample['attention_mask'][:max_tokens_display].tolist()
		print(f"  {mask}")

		# Show special tokens
		print(f"\nSpecial Tokens:")
		print(f"  START token (257) in input_ids: {257 in input_tokens}")
		print(f"  END token (258) in labels:      {258 in label_tokens}")
		print(f"  PAD token (256) present:        {256 in input_tokens or 256 in label_tokens}")

	# Decode tokens back to text
	if show_decoded:
		print(f"\nDecoded Text (from input_ids):")
		print("-" * 80)
		decoded = dataset.tokenizer.decode(sample['input_ids'])
		display_decoded = decoded[:500]
		if len(decoded) > 500:
			display_decoded += "\n... (truncated)"
		print(display_decoded)
		print("-" * 80)

	print()


def validate_dataset(dataset: TGNDataset, num_samples: int = 5):
	"""
	Run validation checks on the dataset.

	Args:
	    dataset: TGNDataset instance
	    num_samples: Number of samples to validate
	"""
	print("\n" + "=" * 80)
	print("DATASET VALIDATION")
	print("=" * 80)

	num_samples = min(num_samples, len(dataset))

	print(f"\nValidating {num_samples} samples...")

	errors = []

	for idx in range(num_samples):
		try:
			sample = dataset[idx]

			# Check tensor types
			assert isinstance(sample['input_ids'], torch.Tensor), f"Sample {idx}: input_ids not a tensor"
			assert isinstance(sample['labels'], torch.Tensor), f"Sample {idx}: labels not a tensor"
			assert isinstance(sample['attention_mask'], torch.Tensor), f"Sample {idx}: attention_mask not a tensor"

			# Check shapes
			assert sample['input_ids'].shape == sample['labels'].shape, f"Sample {idx}: shape mismatch"
			assert sample['input_ids'].shape == sample['attention_mask'].shape, f"Sample {idx}: mask shape mismatch"

			# Check token range
			assert sample['input_ids'].max() < 259, f"Sample {idx}: invalid token ID"
			assert sample['input_ids'].min() >= 0, f"Sample {idx}: negative token ID"

			# Check attention mask values
			assert set(sample['attention_mask'].unique().tolist()).issubset({0, 1}), f"Sample {idx}: invalid mask values"

			# Check sequence structure (input should start with START token)
			first_token = sample['input_ids'][0].item()
			assert first_token == 257, f"Sample {idx}: should start with START token (257), got {first_token}"

		except Exception as e:
			errors.append(f"Sample {idx}: {str(e)}")

	if errors:
		print("\n✗ Validation FAILED:")
		for error in errors:
			print(f"  {error}")
	else:
		print(f"\n✓ Validation PASSED!")
		print(f"  All {num_samples} samples validated successfully")
		print(f"  ✓ Correct tensor types")
		print(f"  ✓ Consistent shapes")
		print(f"  ✓ Valid token ranges")
		print(f"  ✓ Valid attention masks")
		print(f"  ✓ Proper sequence structure")

	print()


def list_samples(dataset: TGNDataset, max_display: int = 20):
	"""
	List all samples in the dataset.

	Args:
	    dataset: TGNDataset instance
	    max_display: Maximum number of samples to display
	"""
	print("\n" + "=" * 80)
	print("DATASET SAMPLES")
	print("=" * 80)

	num_files = len(dataset)
	display_count = min(num_files, max_display)

	print(f"\nShowing {display_count} of {num_files} samples:\n")

	for idx in range(display_count):
		file_info = dataset.get_file_info(idx)
		print(f"  [{idx:3d}] {file_info['name']:<40} ({file_info['size_bytes']:>6} bytes)")

	if num_files > max_display:
		print(f"\n  ... and {num_files - max_display} more samples")

	print()


def visualize_batch_interactive(dataset: TGNDataset, batch_size: int = 4, start_idx: int = 0):
	"""
	Interactive batch visualization with matplotlib.

	Shows visualizations for each batch. Closing the window displays the next batch.
	Press 'q' or close all windows to exit.

	Args:
	    dataset: TGNDataset instance
	    batch_size: Number of samples per batch
	    start_idx: Starting batch index
	"""
	print("\n" + "=" * 80)
	print("INTERACTIVE BATCH VISUALIZATION")
	print("=" * 80)
	print("\nControls:")
	print("  - Close window to view next batch")
	print("  - Press 'q' in window or Ctrl+C to exit")
	print("=" * 80 + "\n")

	# Create DataLoader for batching
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		collate_fn=TGNDataset.collate_batch,
	)

	total_batches = len(dataloader)

	try:
		for batch_idx, batch in enumerate(dataloader):
			if batch_idx < start_idx:
				continue

			visualize_batch(dataset, batch, batch_idx, batch_size, total_batches)

			print(f"\nBatch {batch_idx + 1}/{total_batches} displayed.")
			print("Close the window to view next batch, or press Ctrl+C to exit.\n")

	except KeyboardInterrupt:
		print("\n\nVisualization interrupted by user.")

	print("\nVisualization complete.")


def visualize_batch(
	dataset: TGNDataset,
	batch: dict,
	batch_idx: int,
	batch_size: int,
	total_batches: int,
):
	"""
	Visualize a single batch using matplotlib.

	Creates a comprehensive visualization showing:
	- Input token IDs heatmap
	- Label token IDs heatmap
	- Attention mask
	- Token distribution histogram
	- Sequence statistics

	Args:
	    dataset: TGNDataset instance
	    batch: Batch dictionary with input_ids, labels, attention_mask
	    batch_idx: Current batch index
	    batch_size: Batch size
	    total_batches: Total number of batches
	"""
	input_ids = batch['input_ids']  # [batch_size, seq_len]
	labels = batch['labels']
	attention_mask = batch['attention_mask']

	actual_batch_size = input_ids.shape[0]
	seq_len = input_ids.shape[1]

	# Create figure with subplots
	fig = plt.figure(figsize=(16, 10))
	fig.suptitle(
		f'Batch {batch_idx + 1}/{total_batches} - {actual_batch_size} samples × {seq_len} tokens',
		fontsize=14,
		fontweight='bold',
	)

	# Create grid spec for better layout
	gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

	# 1. Input IDs Heatmap (top left, span 2 columns)
	ax1 = fig.add_subplot(gs[0, :2])
	input_data = input_ids.cpu().numpy()
	im1 = ax1.imshow(input_data, aspect='auto', cmap='viridis', interpolation='nearest')
	ax1.set_title('Input Token IDs', fontsize=12, fontweight='bold')
	ax1.set_xlabel('Token Position')
	ax1.set_ylabel('Sample in Batch')
	ax1.set_yticks(range(actual_batch_size))
	ax1.set_yticklabels([f'Sample {i}' for i in range(actual_batch_size)])
	cbar1 = plt.colorbar(im1, ax=ax1)
	cbar1.set_label('Token ID', rotation=270, labelpad=15)

	# 2. Label IDs Heatmap (middle left, span 2 columns)
	ax2 = fig.add_subplot(gs[1, :2])
	label_data = labels.cpu().numpy()
	im2 = ax2.imshow(label_data, aspect='auto', cmap='plasma', interpolation='nearest')
	ax2.set_title('Label Token IDs (Next Token Targets)', fontsize=12, fontweight='bold')
	ax2.set_xlabel('Token Position')
	ax2.set_ylabel('Sample in Batch')
	ax2.set_yticks(range(actual_batch_size))
	ax2.set_yticklabels([f'Sample {i}' for i in range(actual_batch_size)])
	cbar2 = plt.colorbar(im2, ax=ax2)
	cbar2.set_label('Token ID', rotation=270, labelpad=15)

	# 3. Attention Mask (bottom left, span 2 columns)
	ax3 = fig.add_subplot(gs[2, :2])
	mask_data = attention_mask.cpu().numpy()
	im3 = ax3.imshow(mask_data, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
	ax3.set_title('Attention Mask (1=Valid, 0=Padding)', fontsize=12, fontweight='bold')
	ax3.set_xlabel('Token Position')
	ax3.set_ylabel('Sample in Batch')
	ax3.set_yticks(range(actual_batch_size))
	ax3.set_yticklabels([f'Sample {i}' for i in range(actual_batch_size)])
	cbar3 = plt.colorbar(im3, ax=ax3, ticks=[0, 1])
	cbar3.set_label('Mask Value', rotation=270, labelpad=15)

	# 4. Token Distribution (top right)
	ax4 = fig.add_subplot(gs[0, 2])
	# Flatten and count tokens (exclude padding)
	valid_tokens = input_ids[attention_mask == 1].cpu().numpy()
	token_counts = np.bincount(valid_tokens, minlength=259)

	# Show distribution of most common tokens
	top_k = 20
	top_indices = np.argsort(token_counts)[-top_k:][::-1]
	top_counts = token_counts[top_indices]

	ax4.barh(range(top_k), top_counts, color='steelblue')
	ax4.set_yticks(range(top_k))
	ax4.set_yticklabels([f'Token {idx}' for idx in top_indices], fontsize=8)
	ax4.set_xlabel('Count')
	ax4.set_title(f'Top {top_k} Token Distribution', fontsize=11, fontweight='bold')
	ax4.invert_yaxis()
	ax4.grid(axis='x', alpha=0.3)

	# 5. Sequence Statistics (middle right)
	ax5 = fig.add_subplot(gs[1, 2])
	ax5.axis('off')

	# Calculate statistics
	non_pad_per_sample = attention_mask.sum(dim=1).cpu().numpy()
	pad_per_sample = seq_len - non_pad_per_sample

	stats_text = "SEQUENCE STATISTICS\n" + "=" * 25 + "\n\n"
	stats_text += f"Batch Size: {actual_batch_size}\n"
	stats_text += f"Sequence Length: {seq_len}\n\n"

	stats_text += "Non-Padding Tokens:\n"
	stats_text += f"  Mean: {non_pad_per_sample.mean():.1f}\n"
	stats_text += f"  Min:  {non_pad_per_sample.min()}\n"
	stats_text += f"  Max:  {non_pad_per_sample.max()}\n\n"

	stats_text += "Padding Tokens:\n"
	stats_text += f"  Mean: {pad_per_sample.mean():.1f}\n"
	stats_text += f"  Min:  {pad_per_sample.min()}\n"
	stats_text += f"  Max:  {pad_per_sample.max()}\n\n"

	# Count special tokens
	start_count = (input_ids == 257).sum().item()
	end_count = (labels == 258).sum().item()
	pad_count = (input_ids == 256).sum().item()

	stats_text += "Special Tokens:\n"
	stats_text += f"  START (257): {start_count}\n"
	stats_text += f"  END (258):   {end_count}\n"
	stats_text += f"  PAD (256):   {pad_count}\n"

	ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
	         fontsize=9, verticalalignment='top', fontfamily='monospace',
	         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

	# 6. Sample Length Distribution (bottom right)
	ax6 = fig.add_subplot(gs[2, 2])
	ax6.bar(range(actual_batch_size), non_pad_per_sample, color='coral', alpha=0.7, label='Valid Tokens')
	ax6.bar(range(actual_batch_size), pad_per_sample, bottom=non_pad_per_sample,
	        color='lightgray', alpha=0.7, label='Padding')
	ax6.set_xlabel('Sample Index')
	ax6.set_ylabel('Token Count')
	ax6.set_title('Token Distribution per Sample', fontsize=11, fontweight='bold')
	ax6.set_xticks(range(actual_batch_size))
	ax6.legend(fontsize=8)
	ax6.grid(axis='y', alpha=0.3)

	# Add sample filenames as text below the plot
	if hasattr(dataset, 'get_file_info'):
		sample_info_text = "Samples in this batch:\n"
		start_sample_idx = batch_idx * batch_size
		for i in range(actual_batch_size):
			sample_idx = start_sample_idx + i
			if sample_idx < len(dataset):
				file_info = dataset.get_file_info(sample_idx)
				sample_info_text += f"  [{i}] {file_info['name']} ({file_info['size_bytes']} bytes)\n"

		fig.text(0.5, 0.01, sample_info_text, ha='center', fontsize=8,
		         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

	plt.tight_layout(rect=[0, 0.08, 1, 0.96])

	# Show plot and wait for user to close it
	plt.show(block=True)
	plt.close(fig)


def main():
	"""Main CLI entry point."""
	parser = argparse.ArgumentParser(
		description="View and validate TGNDataset contents",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # View dataset statistics
  python tools/view_dataset.py configs/training/trigo-gpt2.yaml --stats

  # View a specific sample
  python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0

  # List all samples
  python tools/view_dataset.py configs/training/trigo-gpt2.yaml --list

  # Validate dataset
  python tools/view_dataset.py configs/training/trigo-gpt2.yaml --validate

  # View sample with full details
  python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --tokens --decoded

  # Interactive batch visualization
  python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 4
		""",
	)

	parser.add_argument(
		"config",
		type=Path,
		help="Path to training config file (e.g., configs/training/trigo-gpt2.yaml)",
	)

	parser.add_argument(
		"--stats",
		action="store_true",
		help="Display dataset statistics",
	)

	parser.add_argument(
		"--list",
		action="store_true",
		help="List all samples in the dataset",
	)

	parser.add_argument(
		"--sample",
		type=int,
		metavar="IDX",
		help="Display specific sample by index",
	)

	parser.add_argument(
		"--validate",
		action="store_true",
		help="Run validation checks on the dataset",
	)

	parser.add_argument(
		"--validate-samples",
		type=int,
		default=5,
		metavar="N",
		help="Number of samples to validate (default: 5)",
	)

	parser.add_argument(
		"--visualize",
		action="store_true",
		help="Interactive batch visualization with matplotlib",
	)

	parser.add_argument(
		"--batch-size",
		type=int,
		default=4,
		metavar="N",
		help="Batch size for visualization (default: 4)",
	)

	parser.add_argument(
		"--start-batch",
		type=int,
		default=0,
		metavar="N",
		help="Starting batch index for visualization (default: 0)",
	)

	parser.add_argument(
		"--tokens",
		action="store_true",
		help="Show token sequences when displaying samples",
	)

	parser.add_argument(
		"--decoded",
		action="store_true",
		help="Show decoded text from tokens",
	)

	parser.add_argument(
		"--no-text",
		action="store_true",
		help="Hide original text when displaying samples",
	)

	parser.add_argument(
		"--max-tokens",
		type=int,
		default=50,
		metavar="N",
		help="Maximum number of tokens to display (default: 50)",
	)

	args = parser.parse_args()

	# Validate config path
	if not args.config.exists():
		print(f"Error: Config file not found: {args.config}")
		sys.exit(1)

	# Load dataset
	print(f"\nLoading dataset from config: {args.config}")
	try:
		dataset = load_dataset_from_config(args.config)
	except Exception as e:
		print(f"\nError loading dataset: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)

	# If no specific action specified, show stats by default
	if not (args.stats or args.list or args.sample is not None or args.validate or args.visualize):
		args.stats = True

	# Execute requested actions
	if args.visualize:
		# Visualization mode - interactive batch viewing
		visualize_batch_interactive(dataset, batch_size=args.batch_size, start_idx=args.start_batch)

	if args.stats:
		display_dataset_stats(dataset)

	if args.list:
		list_samples(dataset)

	if args.sample is not None:
		if args.sample < 0 or args.sample >= len(dataset):
			print(f"Error: Sample index {args.sample} out of range [0, {len(dataset) - 1}]")
			sys.exit(1)

		display_sample(
			dataset,
			args.sample,
			show_tokens=args.tokens,
			show_text=not args.no_text,
			show_decoded=args.decoded,
			max_tokens_display=args.max_tokens,
		)

	if args.validate:
		validate_dataset(dataset, num_samples=args.validate_samples)


if __name__ == "__main__":
	main()
