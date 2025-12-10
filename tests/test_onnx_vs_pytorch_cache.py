#!/usr/bin/env python3
"""
Compare ONNX cached inference with PyTorch cached inference.

Since PyTorch cached inference matches direct inference (diff=0.000053),
any difference here indicates ONNX export issue.
"""

import sys
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trigor.data.tokenizer import TGNTokenizer


def test_onnx_vs_pytorch_cache():
    """Compare ONNX cached inference with PyTorch cached inference."""

    # Config
    checkpoint_path = Path("/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/checkpoints/ep0042_val_loss_2.4659.chkpt")
    model_dir = Path("/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_shared_cached")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Checkpoint: {checkpoint_path}")
    print(f"ONNX models: {model_dir}")
    print(f"Device: {device}")

    # ================================================================
    # Load PyTorch model
    # ================================================================
    print("\n" + "="*60)
    print("Loading PyTorch Model")
    print("="*60)

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

    print(f"Model loaded: {type(base_model).__name__}")
    print(f"VALUE token ID: {value_id}")

    # ================================================================
    # Load ONNX models
    # ================================================================
    print("\n" + "="*60)
    print("Loading ONNX Models")
    print("="*60)

    prefix_model_path = model_dir / "base_model_prefix.onnx"
    eval_cached_model_path = model_dir / "base_model_eval_cached.onnx"
    value_head_path = model_dir / "value_head.onnx"

    prefix_session = ort.InferenceSession(str(prefix_model_path))
    eval_cached_session = ort.InferenceSession(str(eval_cached_model_path))
    value_session = ort.InferenceSession(str(value_head_path))

    print("ONNX models loaded")

    # ================================================================
    # Tokenize input
    # ================================================================
    tokenizer = TGNTokenizer()

    test_tgn = "[Board 5x5]\n\n1. Pass"
    tokens = tokenizer.encode(test_tgn, max_length=256, add_special_tokens=False, padding=False)

    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()

    # Add START token
    input_tokens = [1] + tokens
    n = len(input_tokens)

    print(f"\nTest TGN: {test_tgn}")
    print(f"Tokens: {input_tokens}")
    print(f"Sequence length (n): {n}")

    # ================================================================
    # PyTorch cached inference
    # ================================================================
    print("\n" + "="*60)
    print("PyTorch Cached Inference")
    print("="*60)

    with torch.no_grad():
        prefix_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

        # Standard causal mask for prefix
        prefix_mask = torch.tril(torch.ones(n, n, device=device))
        prefix_mask = prefix_mask.unsqueeze(0).unsqueeze(0)
        prefix_mask = torch.where(prefix_mask == 1, 0.0, float('-inf'))

        # Compute prefix KV cache
        prefix_outputs = base_model(
            prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
            output_hidden_states=True
        )
        prefix_kv = prefix_outputs.past_key_values
        prefix_hidden = prefix_outputs.hidden_states[-1]

        print(f"PyTorch prefix KV cache: {len(prefix_kv)} layers")
        print(f"  Layer 0 key shape: {prefix_kv[0][0].shape}")
        print(f"  Prefix hidden shape: {prefix_hidden.shape}")

        # Print KV cache values for comparison
        print(f"\nPyTorch KV cache layer 0 key (first 5 values at position 0, head 0):")
        print(f"  {prefix_kv[0][0][0, 0, 0, :5].cpu().numpy()}")

        # VALUE token evaluation with cache
        # IMPORTANT: To match ONNX eval_cached, we need to:
        # 1. Prepend a dummy token (0) at position prefix_length - 1
        # 2. Then VALUE token at position prefix_length + mask_row_sums - 1

        # For single VALUE token with evaluated_mask=[[1.0]], mask_row_sums=1.0
        # Position 0 (dummy): n - 1 = 20
        # Position 1 (VALUE): n + 1 - 1 = 21

        dummy_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        value_token = torch.tensor([[value_id]], dtype=torch.long, device=device)
        input_tokens_eval = torch.cat([dummy_token, value_token], dim=1)  # [[0, 3]]

        # Position IDs: [n-1, n] where n = prefix_length = 21
        position_ids = torch.tensor([[n - 1, n]], dtype=torch.long, device=device)

        # Attention mask: [batch, 1, 2, prefix_length + 2]
        # Row 0 (dummy at pos n-1): attend to all prefix
        # Row 1 (VALUE at pos n): attend to all prefix + dummy
        query_len = 2
        key_len = n + query_len  # Cache + current input
        attention_mask = torch.zeros(1, query_len, key_len, device=device)

        # Row 0 (dummy): attend to all prefix positions
        attention_mask[0, 0, :n] = 1.0

        # Row 1 (VALUE): attend to all prefix + dummy
        attention_mask[0, 1, :n] = 1.0  # All prefix
        attention_mask[0, 1, n] = 1.0   # The dummy token
        # Note: VALUE should NOT attend to itself per evaluated_mask=[[1.0]]
        # But wait, evaluated_mask=[[1.0]] means VALUE attends to itself
        # Let me check the ONNX code again...

        # Looking at exportOnnx.py line 1055-1057:
        # attention_mask[:, 1:, :prefix_length] = 1.0  # All prefix
        # attention_mask[:, 1:, prefix_length] = 1.0  # The dummy prefix last token
        # attention_mask[:, 1:, prefix_length + 1:] = evaluated_mask  # Other evaluated

        # For m=1, evaluated_mask has shape [1,1,1] with value [[1.0]]
        # attention_mask[:, 1:, prefix_length + 1:] means positions n+1:n+2 (only position n+1)
        # But that's the VALUE token itself, so VALUE attends to itself

        # So attention_mask should be:
        # Row 0 (dummy): attend to positions 0:21 (prefix)
        # Row 1 (VALUE): attend to positions 0:21 (prefix), 21 (dummy), 22 (itself per mask)
        # Wait, that's inconsistent. Let me re-check...

        # Actually in the exported ONNX model:
        # key_len = prefix_length + query_len = 21 + 2 = 23
        # Row 1: attention_mask[:, 1, :21] = 1.0 (prefix)
        # Row 1: attention_mask[:, 1, 21] = 1.0 (dummy)
        # Row 1: attention_mask[:, 1, 22:] = evaluated_mask = [[1.0]] -> attend to itself

        attention_mask[0, 1, n + 1] = 1.0  # VALUE attends to itself (from evaluated_mask)

        print(f"\n[DEBUG] Attention mask construction:")
        print(f"  input_tokens_eval: {input_tokens_eval.tolist()}")
        print(f"  position_ids: {position_ids.tolist()}")
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  attention_mask row 0 (dummy): attend to positions 0:{n}")
        print(f"  attention_mask row 1 (VALUE): attend to positions 0:{n}, {n} (dummy), {n+1} (self)")

        attention_mask = torch.where(attention_mask == 1.0, 0.0, float('-inf'))
        attention_mask = attention_mask.unsqueeze(1)  # [1, 1, 2, n+2]

        # Use prefix KV cache (only n positions, not n+1)
        # We need to truncate the cache to match ONNX behavior
        # and convert to DynamicCache for transformers compatibility
        from transformers import DynamicCache
        truncated_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(prefix_kv):
            truncated_cache.update(k[:, :, :n, :], v[:, :, :n, :], layer_idx=layer_idx)

        eval_outputs = base_model(
            input_tokens_eval,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=truncated_cache,
            use_cache=False,
            output_hidden_states=True
        )

        pytorch_hidden = eval_outputs.hidden_states[-1]  # [1, 2, hidden_dim]
        # Position 1 is VALUE token
        pytorch_hidden = pytorch_hidden[:, 1, :]  # [1, hidden_dim]
        pytorch_value = value_head(pytorch_hidden)

        print(f"\nPyTorch VALUE hidden shape: {pytorch_hidden.shape}")
        print(f"PyTorch VALUE hidden first 5: {pytorch_hidden[0, :5].cpu().numpy()}")
        print(f"PyTorch VALUE hidden norm: {pytorch_hidden.norm().item():.6f}")
        print(f"PyTorch VALUE: {pytorch_value.item():.6f}")

    # ================================================================
    # ONNX cached inference
    # ================================================================
    print("\n" + "="*60)
    print("ONNX Cached Inference")
    print("="*60)

    # Step 1: Compute prefix cache
    prefix_ids_np = np.array([input_tokens], dtype=np.int64)
    prefix_outputs_onnx = prefix_session.run(None, {"prefix_ids": prefix_ids_np})

    num_layers = len(prefix_outputs_onnx) // 2

    print(f"ONNX prefix outputs: {len(prefix_outputs_onnx)} tensors ({num_layers} layers)")
    print(f"  Layer 0 key shape: {prefix_outputs_onnx[0].shape}")

    # Print KV cache values for comparison
    print(f"\nONNX KV cache layer 0 key (first 5 values at position 0, head 0):")
    print(f"  {prefix_outputs_onnx[0][0, 0, 0, :5]}")

    # Compare KV cache (only prefix portion since PyTorch cache includes current position)
    pytorch_kv_layer0_key = prefix_kv[0][0].cpu().numpy()[:, :, :n, :]  # Take only prefix portion
    onnx_kv_layer0_key = prefix_outputs_onnx[0]

    print(f"\nKV Cache Comparison (Layer 0 Key, prefix only):")
    print(f"  PyTorch shape: {pytorch_kv_layer0_key.shape}")
    print(f"  ONNX shape: {onnx_kv_layer0_key.shape}")
    if pytorch_kv_layer0_key.shape == onnx_kv_layer0_key.shape:
        kv_diff = np.abs(pytorch_kv_layer0_key - onnx_kv_layer0_key)
        print(f"  Max diff: {kv_diff.max():.6e}")
        print(f"  Mean diff: {kv_diff.mean():.6e}")
    else:
        print(f"  Shapes don't match!")

    # Step 2: Evaluate VALUE token with cache
    eval_ids = np.array([[value_id]], dtype=np.int64)
    # evaluated_mask: For single VALUE token, use [[1.0]] (attends to itself)
    eval_mask = np.array([[[1.0]]], dtype=np.float32)

    eval_inputs = {
        "evaluated_ids": eval_ids,
        "evaluated_mask": eval_mask,
    }
    for i in range(num_layers):
        eval_inputs[f"past_key_{i}"] = prefix_outputs_onnx[i * 2]
        eval_inputs[f"past_value_{i}"] = prefix_outputs_onnx[i * 2 + 1]

    eval_outputs_onnx = eval_cached_session.run(None, eval_inputs)
    onnx_hidden = eval_outputs_onnx[0][:, 0, :]  # [1, hidden_dim]

    print(f"\nONNX VALUE hidden shape: {onnx_hidden.shape}")
    print(f"ONNX VALUE hidden first 5: {onnx_hidden[0, :5]}")
    print(f"ONNX VALUE hidden norm: {np.linalg.norm(onnx_hidden):.6f}")

    # Step 3: Run value head
    onnx_value = value_session.run(None, {"hidden_states": onnx_hidden})[0]
    print(f"ONNX VALUE: {onnx_value[0]:.6f}")

    # ================================================================
    # Hidden state comparison
    # ================================================================
    print("\n" + "="*60)
    print("Hidden State Comparison")
    print("="*60)

    pytorch_hidden_np = pytorch_hidden.cpu().numpy()
    hidden_diff = np.abs(pytorch_hidden_np - onnx_hidden)
    print(f"Max diff: {hidden_diff.max():.6e}")
    print(f"Mean diff: {hidden_diff.mean():.6e}")
    print(f"Relative diff: {hidden_diff.max() / np.abs(pytorch_hidden_np).max():.6e}")

    # ================================================================
    # Value comparison
    # ================================================================
    print("\n" + "="*60)
    print("VALUE Comparison")
    print("="*60)
    print(f"PyTorch VALUE: {pytorch_value.item():+.6f}")
    print(f"ONNX VALUE:    {onnx_value[0]:+.6f}")
    print(f"Difference:    {abs(pytorch_value.item() - onnx_value[0]):.6f}")

    # ================================================================
    # Additional: Compare position IDs handling
    # ================================================================
    print("\n" + "="*60)
    print("Position ID Analysis")
    print("="*60)
    print(f"PyTorch uses position_id = {n} for VALUE token")
    print("ONNX eval_cached model handles position internally")
    print("Check if ONNX model uses correct position IDs...")


if __name__ == "__main__":
    test_onnx_vs_pytorch_cache()
