"""
Test: Dtype vs Sequence Length Impact on Transformer Outputs

This test investigates whether the 3-5% logits difference observed in TreeLM
is caused by:
1. Dtype quantization (bfloat16 numerical precision), OR
2. Sequence length dependency (challenging transformer order-invariance)

Key Question: Are transformers truly order-invariant given fixed position_ids?
"""

import torch
from trigor.models.treeLM import TreeLM
from trigor.models.gpt2CausalLM import GPT2CausalLM
from omegaconf import OmegaConf


def create_model(dtype=torch.float32):
	"""Create GPT2CausalLM with specified dtype"""
	torch.manual_seed(42)  # Reproducible results
	config = OmegaConf.create({
		'vocab_size': 128,
		'hidden_size': 64,
		'num_layers': 2,
		'num_heads': 4,
		'dropout': 0.0,  # Disable for deterministic behavior
	})
	model = GPT2CausalLM.from_config(config)
	model.eval()  # Inference mode

	# Convert to specified dtype
	if dtype != torch.float32:
		model = model.to(dtype=dtype)

	return model


def compute_diff(logits_1, logits_2):
	"""Compute comprehensive difference metrics"""
	diff = logits_1 - logits_2
	return {
		'l2': torch.norm(diff).item(),
		'l1': torch.sum(torch.abs(diff)).item(),
		'mean': torch.mean(torch.abs(diff)).item(),
		'max': torch.max(torch.abs(diff)).item(),
		'rel_error': (torch.norm(diff) / torch.norm(logits_1)).item() if torch.norm(logits_1) > 0 else 0,
	}


def test_token_reordering(dtype):
	"""
	Test 1: Token Reordering with Fixed Position IDs

	Test: [prefix, a, b, c] vs [prefix, a, c, b]
	Position IDs: [0, 1, 2, 3, 4, 4] for both
	Attention mask: Adjusted to match position_ids

	If transformer is order-invariant, diff should be ~0
	"""
	model = create_model(dtype)
	tree_model = TreeLM(model)

	prefix_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
	a, b, c = 40, 50, 60

	# Scenario A: [a, b, c]
	evaluated_ids_A = torch.tensor([[a, b, c]], dtype=torch.long)
	evaluated_mask_A = torch.tensor([
		[[1, 0, 0],  # a sees [a]       -> position 3
		 [1, 1, 0],  # b sees [a, b]    -> position 4
		 [1, 0, 1]]  # c sees [a, c]    -> position 4
	], dtype=torch.float32)

	# Scenario B: [a, c, b] - REORDERED
	evaluated_ids_B = torch.tensor([[a, c, b]], dtype=torch.long)
	evaluated_mask_B = torch.tensor([
		[[1, 0, 0],  # a sees [a]       -> position 3
		 [1, 1, 0],  # c sees [a, c]    -> position 4
		 [1, 0, 1]]  # b sees [a, b]    -> position 4
	], dtype=torch.float32)

	with torch.no_grad():
		logits_A = tree_model(prefix_ids, evaluated_ids_A, evaluated_mask_A)
		logits_B = tree_model(prefix_ids, evaluated_ids_B, evaluated_mask_B)

	# Extract and reorder logits for comparison
	# A: [last_prefix, a, b, c] at indices [0, 1, 2, 3]
	# B: [last_prefix, a, c, b] at indices [0, 1, 2, 3]
	logits_A_a, logits_A_b, logits_A_c = logits_A[0, 1], logits_A[0, 2], logits_A[0, 3]
	logits_B_a, logits_B_c, logits_B_b = logits_B[0, 1], logits_B[0, 2], logits_B[0, 3]

	# Compare: A vs B (after reordering)
	diff_a = compute_diff(logits_A_a, logits_B_a)
	diff_b = compute_diff(logits_A_b, logits_B_b)
	diff_c = compute_diff(logits_A_c, logits_B_c)

	return {
		'dtype': str(dtype).split('.')[-1],
		'diff_a': diff_a,
		'diff_b': diff_b,
		'diff_c': diff_c,
		'max_diff': max(diff_a['max'], diff_b['max'], diff_c['max']),
		'max_rel_error': max(diff_a['rel_error'], diff_b['rel_error'], diff_c['rel_error']),
	}


def test_sequence_padding(dtype):
	"""
	Test 2: Sequence Length with Padded/Masked Tokens

	Test: [prefix, a, b] vs [prefix, a, b, PAD]
	PAD token should be fully masked and not affect results

	If padding is truly neutral, diff should be ~0
	"""
	model = create_model(dtype)
	tree_model = TreeLM(model)

	prefix_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
	a, b = 40, 50
	PAD = 0  # Standard padding token

	# Scenario A: No padding, length 5
	evaluated_ids_A = torch.tensor([[a, b]], dtype=torch.long)
	evaluated_mask_A = torch.tensor([
		[[1, 0],  # a sees [a]
		 [1, 1]]  # b sees [a, b]
	], dtype=torch.float32)

	# Scenario B: With padding, length 6
	evaluated_ids_B = torch.tensor([[a, b, PAD]], dtype=torch.long)
	evaluated_mask_B = torch.tensor([
		[[1, 0, 0],  # a sees [a]        -> PAD masked
		 [1, 1, 0],  # b sees [a, b]     -> PAD masked
		 [0, 0, 0]]  # PAD sees nothing  -> fully masked
	], dtype=torch.float32)

	with torch.no_grad():
		logits_A = tree_model(prefix_ids, evaluated_ids_A, evaluated_mask_A)
		logits_B = tree_model(prefix_ids, evaluated_ids_B, evaluated_mask_B)

	# Compare 'a' and 'b' logits (ignore PAD)
	diff_a = compute_diff(logits_A[0, 1], logits_B[0, 1])
	diff_b = compute_diff(logits_A[0, 2], logits_B[0, 2])

	return {
		'dtype': str(dtype).split('.')[-1],
		'diff_a': diff_a,
		'diff_b': diff_b,
		'max_diff': max(diff_a['max'], diff_b['max']),
		'max_rel_error': max(diff_a['rel_error'], diff_b['rel_error']),
	}


def test_original_scenario(dtype):
	"""
	Test 3: Original Failing Scenario

	Test: [a, b] (length 5) vs [a, b, c] (length 6)
	Token 'c' is masked from 'b', but sequence length differs

	This is the scenario that showed 3-5% difference
	"""
	model = create_model(dtype)
	tree_model = TreeLM(model)

	prefix_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
	a, b, c = 40, 50, 60

	# Scenario 1: [a, b]
	evaluated_ids_1 = torch.tensor([[a, b]], dtype=torch.long)
	evaluated_mask_1 = torch.tensor([
		[[1, 0],  # a sees [a]
		 [1, 1]]  # b sees [a, b]
	], dtype=torch.float32)

	# Scenario 3: [a, b, c]
	evaluated_ids_3 = torch.tensor([[a, b, c]], dtype=torch.long)
	evaluated_mask_3 = torch.tensor([
		[[1, 0, 0],  # a sees [a]
		 [1, 1, 0],  # b sees [a, b]  (c is masked!)
		 [1, 0, 1]]  # c sees [a, c]
	], dtype=torch.float32)

	with torch.no_grad():
		logits_1 = tree_model(prefix_ids, evaluated_ids_1, evaluated_mask_1)
		logits_3 = tree_model(prefix_ids, evaluated_ids_3, evaluated_mask_3)

	# Compare 'a' and 'b' logits
	diff_a = compute_diff(logits_1[0, 1], logits_3[0, 1])
	diff_b = compute_diff(logits_1[0, 2], logits_3[0, 2])

	return {
		'dtype': str(dtype).split('.')[-1],
		'diff_a': diff_a,
		'diff_b': diff_b,
		'max_diff': max(diff_a['max'], diff_b['max']),
		'max_rel_error': max(diff_a['rel_error'], diff_b['rel_error']),
	}


def test_dtype_consistency():
	"""Test 4: Verify dtype consistency in model"""
	model_f32 = create_model(torch.float32)
	model_bf16 = create_model(torch.bfloat16)

	print("Dtype Consistency Check:")
	print(f"  Float32 model dtype: {next(model_f32.parameters()).dtype}")
	print(f"  Float32 pos emb dtype: {model_f32.transformer.wpe.weight.dtype}")
	print(f"  Bfloat16 model dtype: {next(model_bf16.parameters()).dtype}")
	print(f"  Bfloat16 pos emb dtype: {model_bf16.transformer.wpe.weight.dtype}")


def run_all_tests():
	"""Execute all tests and generate comprehensive report"""
	print("=" * 80)
	print("Dtype vs Sequence Length Test Suite")
	print("=" * 80)
	print()

	# Test 4 first: Dtype consistency
	test_dtype_consistency()
	print()

	results = []

	# Test 1: Token reordering
	print("-" * 80)
	print("TEST 1: Token Reordering (Order-Invariance Test)")
	print("-" * 80)
	print("Setup: [a,b,c] vs [a,c,b] with same position_ids [0,1,2,3,4,4]")
	print()

	for dtype in [torch.float32, torch.bfloat16]:
		print(f"Running with {str(dtype).split('.')[-1]}...")
		result = test_token_reordering(dtype)
		results.append(('Test 1: Reordering', result))

		print(f"  Max absolute diff: {result['max_diff']:.6e}")
		print(f"  Max relative error: {result['max_rel_error']:.4%}")
	print()

	# Test 2: Sequence padding
	print("-" * 80)
	print("TEST 2: Sequence Padding (Padding Neutrality Test)")
	print("-" * 80)
	print("Setup: [a,b] vs [a,b,PAD] where PAD is fully masked")
	print()

	for dtype in [torch.float32, torch.bfloat16]:
		print(f"Running with {str(dtype).split('.')[-1]}...")
		result = test_sequence_padding(dtype)
		results.append(('Test 2: Padding', result))

		print(f"  Max absolute diff: {result['max_diff']:.6e}")
		print(f"  Max relative error: {result['max_rel_error']:.4%}")
	print()

	# Test 3: Original scenario
	print("-" * 80)
	print("TEST 3: Original Failing Scenario")
	print("-" * 80)
	print("Setup: [a,b] (len=5) vs [a,b,c] (len=6) where c masked from b")
	print()

	for dtype in [torch.float32, torch.bfloat16]:
		print(f"Running with {str(dtype).split('.')[-1]}...")
		result = test_original_scenario(dtype)
		results.append(('Test 3: Original', result))

		print(f"  Max absolute diff: {result['max_diff']:.6e}")
		print(f"  Max relative error: {result['max_rel_error']:.4%}")
	print()

	# Generate detailed comparison table
	print("=" * 80)
	print("DETAILED RESULTS TABLE")
	print("=" * 80)
	print()

	print(f"{'Test':<25} {'Dtype':<10} {'Token':<8} {'L2':<12} {'Mean':<12} {'Max':<12} {'Rel%':<10}")
	print("-" * 85)

	for test_name, result in results:
		dtype = result['dtype']

		if 'diff_a' in result and 'diff_b' in result:
			# Tests with a and b
			for token, diff in [('a', result['diff_a']), ('b', result['diff_b'])]:
				print(f"{test_name:<25} {dtype:<10} {token:<8} "
					  f"{diff['l2']:<12.6e} {diff['mean']:<12.6e} "
					  f"{diff['max']:<12.6e} {diff['rel_error']:<10.4%}")

			# Add c if present
			if 'diff_c' in result:
				diff = result['diff_c']
				print(f"{test_name:<25} {dtype:<10} {'c':<8} "
					  f"{diff['l2']:<12.6e} {diff['mean']:<12.6e} "
					  f"{diff['max']:<12.6e} {diff['rel_error']:<10.4%}")

	print()
	print("=" * 80)
	print("ANALYSIS & CONCLUSION")
	print("=" * 80)
	print()

	# Analyze Test 1 results
	test1_f32 = [r for r in results if 'Reordering' in r[0] and r[1]['dtype'] == 'float32'][0][1]
	test1_bf16 = [r for r in results if 'Reordering' in r[0] and r[1]['dtype'] == 'bfloat16'][0][1]

	print("Test 1 (Order-Invariance):")
	print(f"  Float32:  max_diff = {test1_f32['max_diff']:.6e}, rel_error = {test1_f32['max_rel_error']:.4%}")
	print(f"  Bfloat16: max_diff = {test1_bf16['max_diff']:.6e}, rel_error = {test1_bf16['max_rel_error']:.4%}")

	if test1_f32['max_diff'] < 1e-6 and test1_bf16['max_diff'] < 1e-2:
		print("  ✓ PASS: Transformer IS order-invariant with fixed position_ids")
	else:
		print("  ⚠ WARNING: Unexpected differences in token reordering!")

	# Analyze Test 2 results
	test2_f32 = [r for r in results if 'Padding' in r[0] and r[1]['dtype'] == 'float32'][0][1]
	test2_bf16 = [r for r in results if 'Padding' in r[0] and r[1]['dtype'] == 'bfloat16'][0][1]

	print()
	print("Test 2 (Padding Neutrality):")
	print(f"  Float32:  max_diff = {test2_f32['max_diff']:.6e}, rel_error = {test2_f32['max_rel_error']:.4%}")
	print(f"  Bfloat16: max_diff = {test2_bf16['max_diff']:.6e}, rel_error = {test2_bf16['max_rel_error']:.4%}")

	if test2_f32['max_diff'] < 1e-4:
		print("  ✓ PASS: Padding is neutral in float32")
	else:
		print("  ⚠ WARNING: Padding affects results even in float32!")
		print("  → Sequence length matters beyond attention mask")

	# Analyze Test 3 results
	test3_f32 = [r for r in results if 'Original' in r[0] and r[1]['dtype'] == 'float32'][0][1]
	test3_bf16 = [r for r in results if 'Original' in r[0] and r[1]['dtype'] == 'bfloat16'][0][1]

	print()
	print("Test 3 (Original Scenario):")
	print(f"  Float32:  max_diff = {test3_f32['max_diff']:.6e}, rel_error = {test3_f32['max_rel_error']:.4%}")
	print(f"  Bfloat16: max_diff = {test3_bf16['max_diff']:.6e}, rel_error = {test3_bf16['max_rel_error']:.4%}")

	print()
	print("-" * 80)
	print("FINAL VERDICT:")
	print("-" * 80)

	# Determine primary cause
	if test3_f32['max_rel_error'] < 0.01:  # < 1%
		print("✓ OUTCOME A: Dtype is the primary cause")
		print("  - Float32 reduces error to <1%")
		print("  - Bfloat16 quantization caused the 3-5% difference")
		print("  - Position_ids fix is CORRECT")
		print("  - Recommendation: Use float32 for critical evaluations")
	elif test3_f32['max_rel_error'] > 0.02:  # > 2%
		print("❌ OUTCOME B: Sequence length is the primary cause")
		print("  - Float32 still shows 2-5% difference")
		print("  - Sequence length inherently affects computation")
		print("  - This CHALLENGES transformer order-invariance assumption!")
		print("  - Recommendation: Investigate transformer architecture")
	else:
		print("⚠ OUTCOME C: Mixed (both contribute)")
		print("  - Float32 shows 1-2% difference")
		print("  - Both dtype and sequence length contribute")
		print("  - Recommendation: Use float32 AND investigate architecture")

	print("=" * 80)


if __name__ == '__main__':
	run_all_tests()
