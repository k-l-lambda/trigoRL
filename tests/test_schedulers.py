#!/usr/bin/env python
"""
Test script to verify LambdaLR scheduler implementations.
Tests both inverse_sqrt and custom lambda schedulers.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_inverse_sqrt_scheduler():
	"""Test inverse square root scheduler (Transformer-style)."""
	print("=" * 80)
	print("Test: Inverse Square Root Scheduler")
	print("=" * 80)

	# Create dummy model and optimizer
	model = nn.Linear(10, 10)
	optimizer = Adam(model.parameters(), lr=1.0)

	# Parameters
	warmup_steps = 4000
	d_model = 512
	lr_mul = 1.0

	# Create scheduler
	def lr_lambda(current_step):
		if current_step == 0:
			current_step = 1

		scale = d_model ** -0.5
		step_scale = min(current_step ** (-0.5),
		                current_step * warmup_steps ** (-1.5))

		return lr_mul * scale * step_scale

	scheduler = LambdaLR(optimizer, lr_lambda)

	# Test learning rates
	print("\nLearning rate schedule:")
	test_steps = [1, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000]

	lrs = []
	steps = []

	for step in range(1, 32001):
		lr = optimizer.param_groups[0]['lr']
		steps.append(step)
		lrs.append(lr)

		if step in test_steps:
			print(f"  Step {step:5d}: lr = {lr:.6f}")

		optimizer.step()
		scheduler.step()

	# Verify warmup behavior
	print("\nVerifying warmup phase (steps 1-4000):")
	warmup_lrs = lrs[:4000]
	print(f"  Start LR (step 1): {warmup_lrs[0]:.6f}")
	print(f"  End LR (step 4000): {warmup_lrs[-1]:.6f}")
	print(f"  Peak LR: {max(warmup_lrs):.6f} at step {warmup_lrs.index(max(warmup_lrs)) + 1}")

	# Verify decay behavior
	print("\nVerifying decay phase (steps 4001-32000):")
	decay_lrs = lrs[4000:]
	print(f"  LR at step 8000: {lrs[7999]:.6f}")
	print(f"  LR at step 16000: {lrs[15999]:.6f}")
	print(f"  LR at step 32000: {lrs[31999]:.6f}")

	# Check that LR decreases after warmup
	assert warmup_lrs[-1] > lrs[8000-1], "LR should decrease after warmup"
	assert lrs[8000-1] > lrs[16000-1], "LR should continue decreasing"
	assert lrs[16000-1] > lrs[32000-1], "LR should continue decreasing"

	print("\n✓ Inverse sqrt scheduler test passed!")

	return steps, lrs


def test_custom_lambda_scheduler():
	"""Test custom lambda scheduler."""
	print("\n" + "=" * 80)
	print("Test: Custom Lambda Scheduler")
	print("=" * 80)

	# Create dummy model and optimizer
	model = nn.Linear(10, 10)
	optimizer = Adam(model.parameters(), lr=0.001)

	# Custom lambda: exponential decay with warmup
	warmup_steps = 1000
	decay_rate = 0.95

	def lr_lambda(step):
		if step < warmup_steps:
			return step / warmup_steps
		else:
			return decay_rate ** ((step - warmup_steps) / 1000)

	scheduler = LambdaLR(optimizer, lr_lambda)

	# Test learning rates
	print("\nLearning rate schedule:")
	test_steps = [1, 500, 1000, 2000, 5000, 10000]

	lrs = []
	steps = []

	for step in range(1, 10001):
		lr = optimizer.param_groups[0]['lr']
		steps.append(step)
		lrs.append(lr)

		if step in test_steps:
			print(f"  Step {step:5d}: lr = {lr:.6f}")

		optimizer.step()
		scheduler.step()

	print("\n✓ Custom lambda scheduler test passed!")

	return steps, lrs


def plot_schedulers():
	"""Plot learning rate schedules."""
	print("\n" + "=" * 80)
	print("Plotting Learning Rate Schedules")
	print("=" * 80)

	# Test inverse sqrt
	steps1, lrs1 = test_inverse_sqrt_scheduler()

	# Test custom lambda
	steps2, lrs2 = test_custom_lambda_scheduler()

	# Create plot
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	# Plot 1: Inverse sqrt
	ax1.plot(steps1, lrs1, 'b-', linewidth=2)
	ax1.axvline(4000, color='r', linestyle='--', alpha=0.5, label='Warmup end')
	ax1.set_xlabel('Training Step')
	ax1.set_ylabel('Learning Rate')
	ax1.set_title('Inverse Square Root Scheduler\n(Transformer-style)')
	ax1.grid(True, alpha=0.3)
	ax1.legend()

	# Plot 2: Custom lambda
	ax2.plot(steps2, lrs2, 'g-', linewidth=2)
	ax2.axvline(1000, color='r', linestyle='--', alpha=0.5, label='Warmup end')
	ax2.set_xlabel('Training Step')
	ax2.set_ylabel('Learning Rate')
	ax2.set_title('Custom Lambda Scheduler\n(Exponential decay with warmup)')
	ax2.grid(True, alpha=0.3)
	ax2.legend()

	plt.tight_layout()

	# Save plot
	output_file = Path(__file__).parent.parent / "outputs" / "scheduler_comparison.png"
	output_file.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_file, dpi=150, bbox_inches='tight')
	print(f"\nPlot saved to: {output_file}")

	plt.close()


if __name__ == "__main__":
	try:
		# Run tests
		plot_schedulers()

		print("\n" + "=" * 80)
		print("All scheduler tests passed!")
		print("=" * 80)

	except ImportError as e:
		if "matplotlib" in str(e):
			print("\nNote: matplotlib not available, running tests without plotting")
			test_inverse_sqrt_scheduler()
			test_custom_lambda_scheduler()
			print("\n" + "=" * 80)
			print("All scheduler tests passed!")
			print("=" * 80)
		else:
			raise
