"""
Test: Pure Token Reordering with Fixed Position IDs

This test verifies whether transformers are truly order-invariant when position_ids are explicitly controlled.

Key Question: If we reorder tokens but maintain position_ids mapping, do we get identical logits?

Expected: If transformer is order-invariant, logits should be IDENTICAL (diff ~1e-7).
"""

import torch
from trigor.models.gpt2CausalLM import GPT2CausalLM
from omegaconf import OmegaConf


def create_model(dtype=torch.float32):
	"""Create GPT2CausalLM with specified dtype"""
	torch.manual_seed(42)
	config = OmegaConf.create({
		'vocab_size': 128,
		'hidden_size': 64,
		'num_layers': 2,
		'num_heads': 4,
		'dropout': 0.0,
	})
	model = GPT2CausalLM.from_config(config)
	model.eval()

	if dtype != torch.float32:
		model = model.to(dtype=dtype)

	return model


def compute_diff(logits_1, logits_2):
	"""Compute difference metrics"""
	diff = logits_1 - logits_2
	return {
		'l2': torch.norm(diff).item(),
		'max': torch.max(torch.abs(diff)).item(),
		'mean': torch.mean(torch.abs(diff)).item(),
		'rel_error': (torch.norm(diff) / torch.norm(logits_1)).item() if torch.norm(logits_1) > 0 else 0,
	}


def test_simple_reordering(dtype):
	"""
	Test 1: Simple 2-token reordering

	Sequence A: [a, b] with position_ids [0, 1]
	Sequence B: [b, a] with position_ids [1, 0]

	Expected: Logits should be identical after reordering back
	"""
	model = create_model(dtype)

	a, b = 40, 50

	# Scenario A: [a, b] + positions [0, 1]
	input_ids_A = torch.tensor([[a, b]], dtype=torch.long)
	position_ids_A = torch.tensor([[0, 1]], dtype=torch.long)
	# Attention mask for A: position-based causal
	# Index 0 (a@pos0) attends to pos0 only → index 0
	# Index 1 (b@pos1) attends to pos0,1 → indices 0,1
	# NOTE: Use log-space mask (0 = attend, -inf = mask) - consistent with transformers
	mask_value = -float("inf")
	attention_mask_A = torch.tensor([
		[[0, mask_value],  # a@pos0 (index 0) attends to itself only
		 [0, 0]]  # b@pos1 (index 1) attends to both
	], dtype=torch.float32).unsqueeze(1)  # Add head dimension

	# Scenario B: [b, a] + positions [1, 0]
	input_ids_B = torch.tensor([[b, a]], dtype=torch.long)
	position_ids_B = torch.tensor([[1, 0]], dtype=torch.long)
	# Attention mask for B: position-based causal (reordered)
	# Index 0 (b@pos1) attends to pos0,1 → indices 0,1
	# Index 1 (a@pos0) attends to pos0 only → index 1
	# NOTE: Use log-space mask (0 = attend, -inf = mask)
	attention_mask_B = torch.tensor([
		[[0, 0],  # b@pos1 (index 0) attends to both (itself and a@pos0)
		 [mask_value, 0]]  # a@pos0 (index 1) attends to itself only
	], dtype=torch.float32).unsqueeze(1)  # Add head dimension

	with torch.no_grad():
		output_A = model(input_ids_A, attention_mask=attention_mask_A, position_ids=position_ids_A)
		output_B = model(input_ids_B, attention_mask=attention_mask_B, position_ids=position_ids_B)

	# Extract logits from model output
	logits_A = output_A.logits if hasattr(output_A, 'logits') else output_A
	logits_B = output_B.logits if hasattr(output_B, 'logits') else output_B

	# Extract logits for comparison
	# A: [a@pos0, b@pos1] at indices [0, 1]
	# B: [b@pos1, a@pos0] at indices [0, 1]
	logits_A_at_pos0 = logits_A[0, 0]  # token 'a' at position 0
	logits_A_at_pos1 = logits_A[0, 1]  # token 'b' at position 1

	logits_B_at_pos1 = logits_B[0, 0]  # token 'b' at position 1 (index 0)
	logits_B_at_pos0 = logits_B[0, 1]  # token 'a' at position 0 (index 1)

	# Compare: position 0 vs position 0, position 1 vs position 1
	diff_pos0 = compute_diff(logits_A_at_pos0, logits_B_at_pos0)
	diff_pos1 = compute_diff(logits_A_at_pos1, logits_B_at_pos1)

	return {
		'dtype': str(dtype).split('.')[-1],
		'diff_pos0': diff_pos0,
		'diff_pos1': diff_pos1,
		'max_diff': max(diff_pos0['max'], diff_pos1['max']),
		'max_rel_error': max(diff_pos0['rel_error'], diff_pos1['rel_error']),
	}


def test_long_sequence_shuffling(dtype):
	"""
	Test 2: Longer sequence with shuffled order

	Sequence A: [a,b,c,d,e,f,g] with position_ids [0,1,2,3,4,5,6]
	Sequence B: [g,d,b,f,a,c,e] with position_ids [6,3,1,5,0,2,4]

	Mapping:
	  A: a→0, b→1, c→2, d→3, e→4, f→5, g→6
	  B: g→6, d→3, b→1, f→5, a→0, c→2, e→4

	Expected: Logits at each position should match
	"""
	model = create_model(dtype)

	a, b, c, d, e, f, g = 40, 50, 60, 70, 80, 90, 100

	# Scenario A: sequential order
	input_ids_A = torch.tensor([[a, b, c, d, e, f, g]], dtype=torch.long)
	position_ids_A = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)
	# Attention mask A: standard causal (position i attends to positions 0..i)
	# NOTE: Use log-space mask (0 = attend, -inf = mask) - consistent with transformers
	mask_value = -float("inf")
	attention_mask_A = torch.tril(torch.ones(7, 7, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
	# Convert 0/1 to 0/-inf format
	attention_mask_A = torch.where(attention_mask_A == 1, torch.tensor(0.0), torch.tensor(mask_value))

	# Scenario B: shuffled order [g,d,b,f,a,c,e]
	# Positions:                  [6,3,1,5,0,2,4]
	input_ids_B = torch.tensor([[g, d, b, f, a, c, e]], dtype=torch.long)
	position_ids_B = torch.tensor([[6, 3, 1, 5, 0, 2, 4]], dtype=torch.long)

	# Attention mask B: position-based causal
	# Each token at position p should attend to all positions 0..p
	# Index 0 (g@pos6) → attends to pos 0,1,2,3,4,5,6 → indices 4,2,5,1,6,X,0 → all indices
	# Index 1 (d@pos3) → attends to pos 0,1,2,3 → indices 4,2,5,1
	# Index 2 (b@pos1) → attends to pos 0,1 → indices 4,2
	# Index 3 (f@pos5) → attends to pos 0,1,2,3,4,5 → indices 4,2,5,1,6,3
	# Index 4 (a@pos0) → attends to pos 0 → index 4
	# Index 5 (c@pos2) → attends to pos 0,1,2 → indices 4,2,5
	# Index 6 (e@pos4) → attends to pos 0,1,2,3,4 → indices 4,2,5,1,6

	# Build attention mask for B based on position relationships
	# NOTE: Use log-space mask (0 = attend, -inf = mask) - consistent with transformers
	attention_mask_B = torch.full((7, 7), mask_value, dtype=torch.float32)
	pos_to_idx_B = {6: 0, 3: 1, 1: 2, 5: 3, 0: 4, 2: 5, 4: 6}  # position → index mapping
	for idx in range(7):
		pos = position_ids_B[0, idx].item()
		# This token at position 'pos' should attend to all positions 0..pos
		for target_pos in range(pos + 1):
			target_idx = pos_to_idx_B[target_pos]
			attention_mask_B[idx, target_idx] = 0.0  # Can attend

	attention_mask_B = attention_mask_B.unsqueeze(0).unsqueeze(0)

	with torch.no_grad():
		output_A = model(input_ids_A, attention_mask=attention_mask_A, position_ids=position_ids_A)
		output_B = model(input_ids_B, attention_mask=attention_mask_B, position_ids=position_ids_B)

	# Extract logits from model output
	logits_A = output_A.logits if hasattr(output_A, 'logits') else output_A
	logits_B = output_B.logits if hasattr(output_B, 'logits') else output_B

	# Extract logits by position
	# A: [a@0, b@1, c@2, d@3, e@4, f@5, g@6] at indices [0,1,2,3,4,5,6]
	# B: [g@6, d@3, b@1, f@5, a@0, c@2, e@4] at indices [0,1,2,3,4,5,6]

	logits_A_by_pos = {
		0: logits_A[0, 0],  # a@0
		1: logits_A[0, 1],  # b@1
		2: logits_A[0, 2],  # c@2
		3: logits_A[0, 3],  # d@3
		4: logits_A[0, 4],  # e@4
		5: logits_A[0, 5],  # f@5
		6: logits_A[0, 6],  # g@6
	}

	logits_B_by_pos = {
		6: logits_B[0, 0],  # g@6
		3: logits_B[0, 1],  # d@3
		1: logits_B[0, 2],  # b@1
		5: logits_B[0, 3],  # f@5
		0: logits_B[0, 4],  # a@0
		2: logits_B[0, 5],  # c@2
		4: logits_B[0, 6],  # e@4
	}

	# Compare each position
	diffs = {}
	for pos in range(7):
		diffs[pos] = compute_diff(logits_A_by_pos[pos], logits_B_by_pos[pos])

	max_diff = max(d['max'] for d in diffs.values())
	max_rel_error = max(d['rel_error'] for d in diffs.values())

	return {
		'dtype': str(dtype).split('.')[-1],
		'position_diffs': diffs,
		'max_diff': max_diff,
		'max_rel_error': max_rel_error,
	}


def test_masked_token_insertion(dtype):
	"""
	Test 3: Inserting a fully masked token

	Sequence A: [a, b] with position_ids [0, 2]
	Sequence B: [a, ZERO, b] with position_ids [0, 1, 2]
	  where ZERO is fully masked (can't see anything, can't be seen)

	Question: Does a fully masked token at position 1 affect the computation?
	Expected: Should NOT affect - logits for a@pos0 and b@pos2 should match
	"""
	model = create_model(dtype)

	a, b = 40, 50
	ZERO = 0  # Masked token
	mask_value = -float("inf")

	# Scenario A: [a, b] without masked token
	input_ids_A = torch.tensor([[a, b]], dtype=torch.long)
	position_ids_A = torch.tensor([[0, 2]], dtype=torch.long)
	# Attention mask A:
	# Index 0 (a@pos0) attends to pos0 → index 0
	# Index 1 (b@pos2) attends to pos0,1,2 → indices 0,1 (but index 1 doesn't exist, so just 0)
	# Actually, b@pos2 should attend to pos0,2 → indices 0,1
	# NOTE: Use log-space mask (0 = attend, -inf = mask) - consistent with transformers
	attention_mask_A = torch.tensor([
		[[0, mask_value],  # a@pos0 attends to itself
		 [0, 0]]  # b@pos2 attends to both a@pos0 and itself
	], dtype=torch.float32).unsqueeze(1)

	# Scenario B: [a, ZERO, b] with ZERO fully masked at position 1
	input_ids_B = torch.tensor([[a, ZERO, b]], dtype=torch.long)
	position_ids_B = torch.tensor([[0, 1, 2]], dtype=torch.long)
	# Attention mask B:
	# Index 0 (a@pos0) attends to pos0 → index 0
	# Index 1 (ZERO@pos1) fully masked → attends to nothing
	# Index 2 (b@pos2) attends to pos0,1,2 BUT pos1 is masked → indices 0,2 only
	# NOTE: Use log-space mask (0 = attend, -inf = mask) - consistent with transformers
	attention_mask_B = torch.tensor([
		[[0, mask_value, mask_value],  # a@pos0 attends to itself
		 [mask_value, mask_value, mask_value],  # ZERO@pos1 fully masked (attends to nothing)
		 [0, mask_value, 0]]  # b@pos2 attends to a@pos0 and itself (skips ZERO)
	], dtype=torch.float32).unsqueeze(1)

	with torch.no_grad():
		output_A = model(input_ids_A, attention_mask=attention_mask_A, position_ids=position_ids_A)
		output_B = model(input_ids_B, attention_mask=attention_mask_B, position_ids=position_ids_B)

	# Extract logits from model output
	logits_A = output_A.logits if hasattr(output_A, 'logits') else output_A
	logits_B = output_B.logits if hasattr(output_B, 'logits') else output_B

	# Compare logits for a@pos0 and b@pos2
	# A: a@pos0 at index 0, b@pos2 at index 1
	# B: a@pos0 at index 0, ZERO@pos1 at index 1, b@pos2 at index 2
	logits_A_a = logits_A[0, 0]  # a@pos0
	logits_A_b = logits_A[0, 1]  # b@pos2

	logits_B_a = logits_B[0, 0]  # a@pos0
	logits_B_zero = logits_B[0, 1]  # ZERO@pos1 (for reference)
	logits_B_b = logits_B[0, 2]  # b@pos2

	diff_a = compute_diff(logits_A_a, logits_B_a)
	diff_b = compute_diff(logits_A_b, logits_B_b)

	return {
		'dtype': str(dtype).split('.')[-1],
		'diff_a': diff_a,
		'diff_b': diff_b,
		'max_diff': max(diff_a['max'], diff_b['max']),
		'max_rel_error': max(diff_a['rel_error'], diff_b['rel_error']),
	}


def test_masked_token_position(dtype):
	"""
	Test 4: Masked token at different physical positions (same sequence length)

	Sequence A: [a, b, PAD] with position_ids [0, 2, 1]
	  a@pos0 (index 0), b@pos2 (index 1), PAD@pos1 (index 2, masked)
	Sequence B: [a, ZERO, b] with position_ids [0, 1, 2]
	  a@pos0 (index 0), ZERO@pos1 (index 1, masked), b@pos2 (index 2)

	Both sequences have:
	- Same length (3)
	- Same position→token mapping: a@pos0, b@pos2, masked@pos1
	- But masked token at different PHYSICAL indices: 2 vs 1

	Question: Does physical index of masked token matter?
	Expected: Should produce identical logits (position_ids + mask should fully determine output)
	"""
	model = create_model(dtype)

	a, b = 40, 50
	PAD = 0   # Masked token at tail
	ZERO = 0  # Masked token in middle
	mask_value = -float("inf")

	# Scenario A: [a, b, PAD] with PAD at end
	input_ids_A = torch.tensor([[a, b, PAD]], dtype=torch.long)
	position_ids_A = torch.tensor([[0, 2, 1]], dtype=torch.long)  # a@pos0, b@pos2, PAD@pos1
	# Attention mask A:
	# Index 0 (a@pos0) attends to pos0 → index 0
	# Index 1 (b@pos2) attends to pos0,1,2 BUT pos1 is masked → indices 0,1 (skips PAD)
	# Index 2 (PAD@pos1) fully masked
	# NOTE: Use log-space mask (0 = attend, -inf = mask) - consistent with transformers
	attention_mask_A = torch.tensor([
		[[0, mask_value, mask_value],  # a@pos0 attends to itself
		 [0, 0, mask_value],  # b@pos2 attends to a@pos0 and itself (PAD masked)
		 [mask_value, mask_value, mask_value]]  # PAD@pos1 fully masked
	], dtype=torch.float32).unsqueeze(1)

	# Scenario B: [a, ZERO, b] with ZERO in middle (same as Test 3 Sequence B)
	input_ids_B = torch.tensor([[a, ZERO, b]], dtype=torch.long)
	position_ids_B = torch.tensor([[0, 1, 2]], dtype=torch.long)
	# Attention mask B:
	# Index 0 (a@pos0) attends to pos0 → index 0
	# Index 1 (ZERO@pos1) fully masked
	# Index 2 (b@pos2) attends to pos0,2 → indices 0,2 (skips ZERO)
	# NOTE: Use log-space mask (0 = attend, -inf = mask)
	attention_mask_B = torch.tensor([
		[[0, mask_value, mask_value],  # a@pos0 attends to itself
		 [mask_value, mask_value, mask_value],  # ZERO@pos1 fully masked
		 [0, mask_value, 0]]  # b@pos2 attends to a@pos0 and itself (skips ZERO)
	], dtype=torch.float32).unsqueeze(1)

	with torch.no_grad():
		output_A = model(input_ids_A, attention_mask=attention_mask_A, position_ids=position_ids_A)
		output_B = model(input_ids_B, attention_mask=attention_mask_B, position_ids=position_ids_B)

	# Extract logits from model output
	logits_A = output_A.logits if hasattr(output_A, 'logits') else output_A
	logits_B = output_B.logits if hasattr(output_B, 'logits') else output_B

	# Compare logits for a@pos0 and b@pos2
	# A: a@pos0 at index 0, b@pos2 at index 1, PAD at index 2
	# B: a@pos0 at index 0, ZERO@pos1 at index 1, b@pos2 at index 2
	logits_A_a = logits_A[0, 0]  # a@pos0
	logits_A_b = logits_A[0, 1]  # b@pos2

	logits_B_a = logits_B[0, 0]  # a@pos0
	logits_B_b = logits_B[0, 2]  # b@pos2

	diff_a = compute_diff(logits_A_a, logits_B_a)
	diff_b = compute_diff(logits_A_b, logits_B_b)

	return {
		'dtype': str(dtype).split('.')[-1],
		'diff_a': diff_a,
		'diff_b': diff_b,
		'max_diff': max(diff_a['max'], diff_b['max']),
		'max_rel_error': max(diff_a['rel_error'], diff_b['rel_error']),
	}


def run_all_tests():
	"""Run all tests and report results"""
	print("=" * 80)
	print("Pure Token Reordering Test Suite")
	print("=" * 80)
	print()
	print("Question: Are transformers truly order-invariant given fixed position_ids?")
	print()

	# Test 1: Simple reordering
	print("-" * 80)
	print("TEST 1: Simple 2-Token Reordering")
	print("-" * 80)
	print("Setup:")
	print("  Sequence A: [a, b] + position_ids [0, 1]")
	print("  Sequence B: [b, a] + position_ids [1, 0]")
	print()

	results_test1 = []
	for dtype in [torch.float32, torch.bfloat16]:
		print(f"Running with {str(dtype).split('.')[-1]}...")
		result = test_simple_reordering(dtype)
		results_test1.append(result)

		print(f"  Max absolute diff: {result['max_diff']:.6e}")
		print(f"  Max relative error: {result['max_rel_error']:.4%}")
	print()

	# Test 2: Long sequence shuffling
	print("-" * 80)
	print("TEST 2: Long Sequence Shuffling (7 tokens)")
	print("-" * 80)
	print("Setup:")
	print("  Sequence A: [a,b,c,d,e,f,g] + position_ids [0,1,2,3,4,5,6]")
	print("  Sequence B: [g,d,b,f,a,c,e] + position_ids [6,3,1,5,0,2,4]")
	print("  (Same position→token mapping, different order)")
	print()

	results_test2 = []
	for dtype in [torch.float32, torch.bfloat16]:
		print(f"Running with {str(dtype).split('.')[-1]}...")
		result = test_long_sequence_shuffling(dtype)
		results_test2.append(result)

		print(f"  Max absolute diff: {result['max_diff']:.6e}")
		print(f"  Max relative error: {result['max_rel_error']:.4%}")
	print()

	# Test 3: Masked token insertion
	print("-" * 80)
	print("TEST 3: Masked Token Insertion")
	print("-" * 80)
	print("Setup:")
	print("  Sequence A: [a, b] + position_ids [0, 2]")
	print("  Sequence B: [a, ZERO, b] + position_ids [0, 1, 2]")
	print("  ZERO token at position 1 is fully masked (invisible)")
	print()

	results_test3 = []
	for dtype in [torch.float32, torch.bfloat16]:
		print(f"Running with {str(dtype).split('.')[-1]}...")
		result = test_masked_token_insertion(dtype)
		results_test3.append(result)

		print(f"  Max absolute diff: {result['max_diff']:.6e}")
		print(f"  Max relative error: {result['max_rel_error']:.4%}")
	print()

	# Test 4: Masked token position
	print("-" * 80)
	print("TEST 4: Masked Token at Different Physical Indices")
	print("-" * 80)
	print("Setup:")
	print("  Sequence A: [a, b, PAD] + position_ids [0, 2, 1]")
	print("  Sequence B: [a, ZERO, b] + position_ids [0, 1, 2]")
	print("  Same position mapping (a@0, b@2, masked@1), different physical indices")
	print()

	results_test4 = []
	for dtype in [torch.float32, torch.bfloat16]:
		print(f"Running with {str(dtype).split('.')[-1]}...")
		result = test_masked_token_position(dtype)
		results_test4.append(result)

		print(f"  Max absolute diff: {result['max_diff']:.6e}")
		print(f"  Max relative error: {result['max_rel_error']:.4%}")
	print()

	# Detailed results
	print("=" * 80)
	print("DETAILED RESULTS")
	print("=" * 80)
	print()

	print("Test 1 - Simple Reordering:")
	print(f"  {'Dtype':<10} {'Position':<10} {'Max Diff':<15} {'Rel Error':<12}")
	print("-" * 50)
	for result in results_test1:
		dtype = result['dtype']
		for pos in [0, 1]:
			diff = result[f'diff_pos{pos}']
			print(f"  {dtype:<10} {pos:<10} {diff['max']:<15.6e} {diff['rel_error']:<12.4%}")
	print()

	print("Test 2 - Long Sequence:")
	print(f"  {'Dtype':<10} {'Position':<10} {'Max Diff':<15} {'Rel Error':<12}")
	print("-" * 50)
	for result in results_test2:
		dtype = result['dtype']
		for pos in range(7):
			diff = result['position_diffs'][pos]
			print(f"  {dtype:<10} {pos:<10} {diff['max']:<15.6e} {diff['rel_error']:<12.4%}")
	print()

	print("Test 3 - Masked Token Insertion:")
	print(f"  {'Dtype':<10} {'Token':<10} {'Max Diff':<15} {'Rel Error':<12}")
	print("-" * 50)
	for result in results_test3:
		dtype = result['dtype']
		for token, diff in [('a', result['diff_a']), ('b', result['diff_b'])]:
			print(f"  {dtype:<10} {token:<10} {diff['max']:<15.6e} {diff['rel_error']:<12.4%}")
	print()

	print("Test 4 - Masked Token Position:")
	print(f"  {'Dtype':<10} {'Token':<10} {'Max Diff':<15} {'Rel Error':<12}")
	print("-" * 50)
	for result in results_test4:
		dtype = result['dtype']
		for token, diff in [('a', result['diff_a']), ('b', result['diff_b'])]:
			print(f"  {dtype:<10} {token:<10} {diff['max']:<15.6e} {diff['rel_error']:<12.4%}")
	print()

	# Conclusion
	print("=" * 80)
	print("CONCLUSION")
	print("=" * 80)
	print()

	test1_f32 = results_test1[0]
	test1_bf16 = results_test1[1]
	test2_f32 = results_test2[0]
	test2_bf16 = results_test2[1]
	test3_f32 = results_test3[0]
	test3_bf16 = results_test3[1]
	test4_f32 = results_test4[0]
	test4_bf16 = results_test4[1]

	tolerance = 1e-5

	print("Test 1 (Simple Reordering):")
	print(f"  Float32:  max_diff = {test1_f32['max_diff']:.6e}, rel_error = {test1_f32['max_rel_error']:.4%}")
	print(f"  Bfloat16: max_diff = {test1_bf16['max_diff']:.6e}, rel_error = {test1_bf16['max_rel_error']:.4%}")

	if test1_f32['max_diff'] < tolerance and test1_bf16['max_diff'] < tolerance:
		print(f"  ✓ PASS: Reordering produces identical results (tolerance={tolerance})")
	else:
		print(f"  ❌ FAIL: Reordering produces different results!")

	print()
	print("Test 2 (Long Sequence Shuffling):")
	print(f"  Float32:  max_diff = {test2_f32['max_diff']:.6e}, rel_error = {test2_f32['max_rel_error']:.4%}")
	print(f"  Bfloat16: max_diff = {test2_bf16['max_diff']:.6e}, rel_error = {test2_bf16['max_rel_error']:.4%}")

	if test2_f32['max_diff'] < tolerance and test2_bf16['max_diff'] < tolerance:
		print(f"  ✓ PASS: Shuffling produces identical results (tolerance={tolerance})")
	else:
		print(f"  ❌ FAIL: Shuffling produces different results!")

	print()
	print("Test 3 (Masked Token Insertion):")
	print(f"  Float32:  max_diff = {test3_f32['max_diff']:.6e}, rel_error = {test3_f32['max_rel_error']:.4%}")
	print(f"  Bfloat16: max_diff = {test3_bf16['max_diff']:.6e}, rel_error = {test3_bf16['max_rel_error']:.4%}")

	if test3_f32['max_diff'] < tolerance and test3_bf16['max_diff'] < tolerance:
		print(f"  ✓ PASS: Masked token is truly invisible (tolerance={tolerance})")
	else:
		print(f"  ❌ FAIL: Masked token affects computation (sequence length matters)!")

	print()
	print("Test 4 (Masked Token Position):")
	print(f"  Float32:  max_diff = {test4_f32['max_diff']:.6e}, rel_error = {test4_f32['max_rel_error']:.4%}")
	print(f"  Bfloat16: max_diff = {test4_bf16['max_diff']:.6e}, rel_error = {test4_bf16['max_rel_error']:.4%}")

	if test4_f32['max_diff'] < tolerance and test4_bf16['max_diff'] < tolerance:
		print(f"  ✓ PASS: Masked token position doesn't matter (tolerance={tolerance})")
	else:
		print(f"  ❌ FAIL: Masked token position affects computation!")

	print()
	print("-" * 80)
	print("VERDICT:")
	print("-" * 80)

	if (test1_f32['max_diff'] < tolerance and test2_f32['max_diff'] < tolerance and
	    test3_f32['max_diff'] < tolerance and test4_f32['max_diff'] < tolerance):
		print("✓ Transformers ARE fully order-invariant with fixed position_ids")
		print("  - Token reordering produces identical logits")
		print("  - Position_ids fully control positional information")
		print("  - Fully masked tokens are truly invisible")
		print("  - Masked token position doesn't matter")
		print("  - This validates the core transformer assumption")
	elif (test1_f32['max_diff'] < tolerance and test2_f32['max_diff'] < tolerance and test4_f32['max_diff'] < tolerance):
		print("✓ Transformers ARE order-invariant with same sequence length")
		print("  - Token reordering produces identical logits (Tests 1 & 2)")
		print("  - Masked token position doesn't matter when length is same (Test 4)")
		print("  ⚠️  BUT: Different sequence lengths produce different results (Test 3)")
		print("  - This is a fundamental property of transformers")
	else:
		print("❌ Transformers are NOT purely order-invariant!")
		print("  - Token reordering produces different logits even with fixed position_ids")
		print("  - This challenges the fundamental transformer assumption")
		print("  - Further investigation needed into transformer architecture")

	print("=" * 80)


if __name__ == '__main__':
	run_all_tests()
