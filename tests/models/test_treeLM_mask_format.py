"""
Test: TreeLM Mask Format Conversion

This test verifies that TreeLM correctly converts 0/1 mask to log-space format (-inf/0).
"""

import torch
from trigor.models.treeLM import TreeLM
from trigor.models.gpt2CausalLM import GPT2CausalLM
from omegaconf import OmegaConf


def test_mask_format_conversion():
	"""Test that TreeLM converts mask to log-space format"""

	# Create small model
	torch.manual_seed(42)
	config = OmegaConf.create({
		'vocab_size': 128,
		'hidden_size': 64,
		'num_layers': 2,
		'num_heads': 4,
		'dropout': 0.0,
	})
	base_model = GPT2CausalLM.from_config(config)
	tree_model = TreeLM(base_model)
	tree_model.eval()

	# Test case: simple 2-token evaluation with custom mask
	prefix_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)  # n=3
	evaluated_ids = torch.tensor([[10, 20]], dtype=torch.long)  # m=2

	# Binary mask (1 = attend, 0 = mask)
	# Token 0 (10) attends to itself only
	# Token 1 (20) attends to both
	evaluated_mask = torch.tensor([
		[[1, 0],   # 10 sees only itself
		 [1, 1]]   # 20 sees both
	], dtype=torch.float32)

	print("=" * 80)
	print("TreeLM Mask Format Conversion Test")
	print("=" * 80)
	print()
	print("Input mask (binary format, 1=attend, 0=mask):")
	print(evaluated_mask[0])
	print()

	# Run forward pass
	with torch.no_grad():
		# Patch the forward method to capture the actual mask
		# TreeLM unwraps to base model, which is GPT2LMHeadModel
		if hasattr(base_model, 'model'):
			target_model = base_model.model
		else:
			target_model = base_model

		original_forward = target_model.forward
		captured_mask = None

		def capture_mask_forward(input_ids, attention_mask=None, **kwargs):
			nonlocal captured_mask
			captured_mask = attention_mask
			return original_forward(input_ids, attention_mask=attention_mask, **kwargs)

		target_model.forward = capture_mask_forward

		try:
			output = tree_model(prefix_ids, evaluated_ids, evaluated_mask)
			print("✓ Forward pass completed successfully")
			print()
		finally:
			target_model.forward = original_forward

	# Verify captured mask
	if captured_mask is None:
		print("❌ FAIL: No mask was captured!")
		return False

	print("Captured mask shape:", captured_mask.shape)
	print("Expected shape: [batch=1, heads=1, seq_len=5, seq_len=5]")
	print()

	# Extract the bottom-right 2x2 region (evaluated region)
	mask_2d = captured_mask[0, 0]  # Remove batch and head dimensions
	evaluated_region = mask_2d[-2:, -2:]  # Last 2x2 region

	print("Captured mask in evaluated region (should be log-space):")
	print(evaluated_region)
	print()

	# Check if mask values are in log-space format
	# Should have 0 for attend and large negative for mask
	mask_values = evaluated_region.unique()
	print("Unique values in mask:", mask_values.tolist())
	print()

	# Expected: 0 for attend, large negative (~ -inf) for mask
	has_zero = torch.any(torch.abs(mask_values) < 1e-5).item()
	has_large_negative = torch.any(mask_values < -1000).item()

	if has_zero and has_large_negative:
		print("✓ PASS: Mask is in log-space format (0, -inf)")
		print(f"  - Found 0 values: {has_zero}")
		print(f"  - Found large negative values (<-1000): {has_large_negative}")

		# Verify specific positions
		print()
		print("Verification of specific positions:")
		print(f"  Position [0,0] (attend): {evaluated_region[0,0].item():.1f} (should be ~0)")
		print(f"  Position [0,1] (mask):   {evaluated_region[0,1].item():.1f} (should be -inf)")
		print(f"  Position [1,0] (attend): {evaluated_region[1,0].item():.1f} (should be ~0)")
		print(f"  Position [1,1] (attend): {evaluated_region[1,1].item():.1f} (should be ~0)")

		return True
	else:
		print("❌ FAIL: Mask is NOT in log-space format!")
		print(f"  - Has 0 values: {has_zero}")
		print(f"  - Has large negative values: {has_large_negative}")
		return False


def test_mask_effect_on_attention():
	"""Test that log-space mask actually blocks attention"""

	# Create small model
	torch.manual_seed(42)
	config = OmegaConf.create({
		'vocab_size': 128,
		'hidden_size': 64,
		'num_layers': 2,
		'num_heads': 4,
		'dropout': 0.0,
	})
	base_model = GPT2CausalLM.from_config(config)
	tree_model = TreeLM(base_model)
	tree_model.eval()

	prefix_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
	# Use different tokens to ensure attention matters
	evaluated_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)  # 3 tokens

	print()
	print("=" * 80)
	print("Mask Effect Test: Verify masked positions don't affect output")
	print("=" * 80)
	print()

	# Case 1: Token 2 can see all tokens (0, 1, 2)
	mask_full = torch.tensor([
		[[1, 0, 0],  # Token 0 sees only itself
		 [1, 1, 0],  # Token 1 sees 0,1
		 [1, 1, 1]]  # Token 2 sees all 0,1,2
	], dtype=torch.float32)

	# Case 2: Token 2 can only see tokens 0 and 2 (1 is masked)
	mask_partial = torch.tensor([
		[[1, 0, 0],  # Token 0 sees only itself
		 [1, 1, 0],  # Token 1 sees 0,1
		 [1, 0, 1]]  # Token 2 sees only 0,2 (1 is masked!)
	], dtype=torch.float32)

	with torch.no_grad():
		output_full = tree_model(prefix_ids, evaluated_ids, mask_full)
		output_partial = tree_model(prefix_ids, evaluated_ids, mask_partial)

	# Output for first two tokens should be identical (same attention pattern)
	diff_token0 = torch.max(torch.abs(output_full[0, 0] - output_partial[0, 0])).item()
	diff_token1 = torch.max(torch.abs(output_full[0, 1] - output_partial[0, 1])).item()

	# Output for third token should be different (different attention patterns)
	diff_token2 = torch.max(torch.abs(output_full[0, 2] - output_partial[0, 2])).item()

	print("Attention patterns:")
	print("  Full mask:    Token 2 attends to [0, 1, 2]")
	print("  Partial mask: Token 2 attends to [0, _, 2] (1 is masked)")
	print()
	print(f"Token 0 output difference: {diff_token0:.6e} (should be ~0)")
	print(f"Token 1 output difference: {diff_token1:.6e} (should be ~0)")
	print(f"Token 2 output difference: {diff_token2:.6e} (should be >0)")
	print()

	if diff_token0 < 1e-5 and diff_token1 < 1e-5 and diff_token2 > 1e-4:
		print("✓ PASS: Mask correctly controls attention")
		print("  - Unaffected tokens have identical output")
		print("  - Affected token has different output")
		return True
	else:
		print("❌ FAIL: Mask not working correctly")
		print(f"  - Token 0 same: {diff_token0 < 1e-5}")
		print(f"  - Token 1 same: {diff_token1 < 1e-5}")
		print(f"  - Token 2 different: {diff_token2 > 1e-4}")
		return False


if __name__ == '__main__':
	print()
	result1 = test_mask_format_conversion()
	result2 = test_mask_effect_on_attention()

	print()
	print("=" * 80)
	print("SUMMARY")
	print("=" * 80)
	if result1 and result2:
		print("✅ All tests passed!")
		print("   TreeLM correctly converts mask to log-space format")
		print("   Masked positions properly block attention")
	else:
		print("❌ Some tests failed!")
		if not result1:
			print("   - Mask format conversion failed")
		if not result2:
			print("   - Mask effect on attention failed")
	print("=" * 80)
	print()
