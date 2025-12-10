#!/usr/bin/env python3
"""
Test different approaches to VALUE inference with prefix cache.

We know:
1. Python checkpoint with direct inference works (baseline)
2. Python checkpoint with prefix cache works (diff=0.000053)
3. ONNX models have discrepancy

This test will systematically compare approaches to identify the issue.
"""

import sys
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trigor.data.tokenizer import TGNTokenizer


def test_approaches():
    """Test different VALUE inference approaches."""

    checkpoint_path = Path("/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/checkpoints/ep0042_val_loss_2.4659.chkpt")
    model_dir = Path("/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']
    if isinstance(config, dict):
        from omegaconf import OmegaConf
        config = OmegaConf.create(config)

    from trigor.models.valueCausalLoss import ValueCausalLoss
    model = ValueCausalLoss.from_config(config.model.config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    base_model = model.model
    value_head = model.value_head
    value_id = model.value_id

    # Tokenize
    tokenizer = TGNTokenizer()
    test_tgn = "[Board 5x5]\n\n1. Pass"
    tokens = tokenizer.encode(test_tgn, max_length=256, add_special_tokens=False, padding=False)
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    input_tokens = [1] + tokens
    n = len(input_tokens)

    print(f"Test TGN: {test_tgn}")
    print(f"Tokens: {input_tokens}")
    print(f"Sequence length (n): {n}")

    # ================================================================
    # Approach 1: Direct (EvaluationLM style) - BASELINE
    # ================================================================
    print("\n" + "="*60)
    print("Approach 1: Direct (EvaluationLM) - BASELINE")
    print("="*60)

    with torch.no_grad():
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
        value_token = torch.full((1, 1), value_id, dtype=torch.long, device=device)
        input_with_value = torch.cat([input_ids, value_token], dim=1)

        outputs = base_model(input_with_value, output_hidden_states=True)
        hidden_direct = outputs.hidden_states[-1][:, -1, :]  # VALUE position
        value_direct = value_head(hidden_direct)

        print(f"VALUE: {value_direct.item():.6f}")
        print(f"Hidden first 5: {hidden_direct[0, :5].cpu().numpy()}")

    # ================================================================
    # Approach 2: Cache + Direct VALUE (no dummy)
    # ================================================================
    print("\n" + "="*60)
    print("Approach 2: Cache + Direct VALUE (no dummy)")
    print("="*60)

    with torch.no_grad():
        prefix_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

        # Compute prefix cache
        prefix_outputs = base_model(
            prefix_ids,
            use_cache=True,
            output_hidden_states=True
        )
        prefix_kv = prefix_outputs.past_key_values

        # VALUE token directly
        value_token = torch.tensor([[value_id]], dtype=torch.long, device=device)

        # Position ID: n (next position after prefix)
        position_ids = torch.tensor([[n]], dtype=torch.long, device=device)

        # Simple approach: let model handle attention mask
        eval_outputs = base_model(
            value_token,
            position_ids=position_ids,
            past_key_values=prefix_kv,
            use_cache=False,
            output_hidden_states=True
        )

        hidden_cache_direct = eval_outputs.hidden_states[-1][:, 0, :]
        value_cache_direct = value_head(hidden_cache_direct)

        print(f"VALUE: {value_cache_direct.item():.6f}")
        print(f"Hidden first 5: {hidden_cache_direct[0, :5].cpu().numpy()}")
        print(f"Diff from baseline: {abs(value_direct.item() - value_cache_direct.item()):.6f}")

    # ================================================================
    # Approach 3: Cache + Dummy + VALUE (ONNX style)
    # ================================================================
    print("\n" + "="*60)
    print("Approach 3: Cache + Dummy + VALUE (ONNX style)")
    print("="*60)

    with torch.no_grad():
        from transformers import DynamicCache

        # Use fresh prefix cache
        prefix_outputs = base_model(
            prefix_ids,
            use_cache=True,
            output_hidden_states=False
        )
        prefix_kv = prefix_outputs.past_key_values

        # Truncate to exactly n positions (matching ONNX behavior)
        truncated_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(prefix_kv):
            truncated_cache.update(k[:, :, :n, :], v[:, :, :n, :], layer_idx=layer_idx)

        # Dummy + VALUE tokens
        dummy_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        value_token = torch.tensor([[value_id]], dtype=torch.long, device=device)
        input_tokens_eval = torch.cat([dummy_token, value_token], dim=1)

        # Position IDs: [n-1, n]
        position_ids = torch.tensor([[n - 1, n]], dtype=torch.long, device=device)

        eval_outputs = base_model(
            input_tokens_eval,
            position_ids=position_ids,
            past_key_values=truncated_cache,
            use_cache=False,
            output_hidden_states=True
        )

        hidden_cache_dummy = eval_outputs.hidden_states[-1][:, 1, :]  # VALUE position
        value_cache_dummy = value_head(hidden_cache_dummy)

        print(f"VALUE: {value_cache_dummy.item():.6f}")
        print(f"Hidden first 5: {hidden_cache_dummy[0, :5].cpu().numpy()}")
        print(f"Diff from baseline: {abs(value_direct.item() - value_cache_dummy.item()):.6f}")

    # ================================================================
    # ONNX inference
    # ================================================================
    print("\n" + "="*60)
    print("ONNX Cached Inference")
    print("="*60)

    prefix_session = ort.InferenceSession(str(model_dir / "base_model_prefix.onnx"))
    eval_cached_session = ort.InferenceSession(str(model_dir / "base_model_eval_cached.onnx"))
    value_session = ort.InferenceSession(str(model_dir / "value_head.onnx"))

    prefix_ids_np = np.array([input_tokens], dtype=np.int64)
    prefix_outputs_onnx = prefix_session.run(None, {"prefix_ids": prefix_ids_np})
    num_layers = len(prefix_outputs_onnx) // 2

    eval_ids = np.array([[value_id]], dtype=np.int64)
    eval_mask = np.array([[[1.0]]], dtype=np.float32)

    eval_inputs = {
        "evaluated_ids": eval_ids,
        "evaluated_mask": eval_mask,
    }
    for i in range(num_layers):
        eval_inputs[f"past_key_{i}"] = prefix_outputs_onnx[i * 2]
        eval_inputs[f"past_value_{i}"] = prefix_outputs_onnx[i * 2 + 1]

    eval_outputs_onnx = eval_cached_session.run(None, eval_inputs)
    # Output shape should be [1, m+1, hidden_dim] or [1, m, hidden_dim] depending on model
    print(f"ONNX eval output shape: {eval_outputs_onnx[0].shape}")

    # Check output shape to determine which position is VALUE
    if eval_outputs_onnx[0].shape[1] == 2:
        # [1, 2, hidden_dim] - position 0 is dummy, position 1 is VALUE
        onnx_hidden = eval_outputs_onnx[0][:, 1, :]
        print("Using position 1 (after dummy)")
    else:
        # [1, 1, hidden_dim] - position 0 is VALUE (no dummy in output)
        onnx_hidden = eval_outputs_onnx[0][:, 0, :]
        print("Using position 0 (no dummy in output)")

    onnx_value = value_session.run(None, {"hidden_states": onnx_hidden})[0]

    print(f"VALUE: {onnx_value[0]:.6f}")
    print(f"Hidden first 5: {onnx_hidden[0, :5]}")
    print(f"Diff from baseline: {abs(value_direct.item() - onnx_value[0]):.6f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline (direct):       {value_direct.item():+.6f}")
    print(f"Cache + Direct VALUE:    {value_cache_direct.item():+.6f} (diff: {abs(value_direct.item() - value_cache_direct.item()):.6f})")
    print(f"Cache + Dummy + VALUE:   {value_cache_dummy.item():+.6f} (diff: {abs(value_direct.item() - value_cache_dummy.item()):.6f})")
    print(f"ONNX cached:             {onnx_value[0]:+.6f} (diff: {abs(value_direct.item() - onnx_value[0]):.6f})")


if __name__ == "__main__":
    test_approaches()
