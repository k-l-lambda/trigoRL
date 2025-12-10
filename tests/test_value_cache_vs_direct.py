#!/usr/bin/env python3
"""
Compare value inference between:
1. EvaluationLM (direct): input_ids → model → hidden[-1] → value_head → value
2. Prefix cache mode: prefix → cache, VALUE → eval_cached → hidden → value_head → value

Goal: Find where the value inference diverges when using prefix cache.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trigor.data.tokenizer import TGNTokenizer


def test_value_inference_comparison():
    """Compare value inference between direct and cached modes."""

    # Config
    checkpoint_path = Path("/home/camus/work/trigoRL/outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/checkpoints/ep0042_val_loss_2.4659.chkpt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']

    # Handle dict config
    if isinstance(config, dict):
        from omegaconf import OmegaConf
        config = OmegaConf.create(config)

    # Create model
    from trigor.models.valueCausalLoss import ValueCausalLoss
    model = ValueCausalLoss.from_config(config.model.config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"Model loaded: {type(model.model).__name__}")

    # Get base model and value head
    base_model = model.model
    value_head = model.value_head
    value_id = model.value_id  # Should be 3

    print(f"VALUE token ID: {value_id}")

    # Tokenizer
    tokenizer = TGNTokenizer()

    # Test input: position after Black Pass
    test_tgn = "[Board 5x5]\n\n1. Pass"
    tokens = tokenizer.encode(test_tgn)

    # Convert to list if tensor
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()

    # Add START token
    input_ids = torch.tensor([[1] + tokens], dtype=torch.long, device=device)
    print(f"\nTest TGN: {test_tgn}")
    print(f"Input tokens: {input_ids.tolist()}")
    print(f"Sequence length: {input_ids.shape[1]}")

    # ================================================================
    # Method 1: EvaluationLM style (direct, no cache)
    # ================================================================
    print("\n" + "="*60)
    print("Method 1: EvaluationLM (Direct, no cache)")
    print("="*60)

    with torch.no_grad():
        # Append VALUE token
        value_token = torch.full((1, 1), value_id, dtype=torch.long, device=device)
        input_with_value = torch.cat([input_ids, value_token], dim=1)

        print(f"Input with VALUE: {input_with_value.shape}")

        # Forward through base model
        outputs = base_model(input_with_value, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer

        print(f"Hidden states shape: {hidden_states.shape}")

        # Get VALUE position hidden state
        value_hidden_direct = hidden_states[:, -1, :]  # [1, hidden_dim]
        print(f"VALUE hidden (direct): shape={value_hidden_direct.shape}")
        print(f"  First 5 dims: {value_hidden_direct[0, :5].tolist()}")
        print(f"  Norm: {value_hidden_direct.norm().item():.6f}")

        # Value head
        value_direct = value_head(value_hidden_direct)
        print(f"VALUE (direct): {value_direct.item():.6f}")

    # ================================================================
    # Method 2: Prefix cache style (simulating eval_cached)
    # ================================================================
    print("\n" + "="*60)
    print("Method 2: Prefix Cache (eval_cached simulation)")
    print("="*60)

    with torch.no_grad():
        # Step 1: Compute prefix cache
        prefix_ids = input_ids  # [1, n]
        n = prefix_ids.shape[1]

        print(f"Prefix length: {n}")

        # Standard causal mask for prefix
        prefix_mask = torch.tril(torch.ones(n, n, device=device))
        prefix_mask = prefix_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
        # Convert to log-space (0 for attend, -inf for mask)
        prefix_mask = torch.where(prefix_mask == 1, 0.0, float('-inf'))

        prefix_outputs = base_model(
            prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
            output_hidden_states=True
        )

        prefix_kv = prefix_outputs.past_key_values
        prefix_hidden = prefix_outputs.hidden_states[-1]

        print(f"Prefix KV cache: {len(prefix_kv)} layers")
        print(f"  Layer 0 key shape: {prefix_kv[0][0].shape}")
        print(f"Prefix hidden shape: {prefix_hidden.shape}")
        print(f"  Last position hidden (prefix n-1): {prefix_hidden[0, -1, :5].tolist()}")

        # Step 2: VALUE token evaluation with cache (eval_cached mode)
        # Following the exact logic from exportOnnx.py

        # Create dummy token for prefix last position
        dummy_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        value_token = torch.tensor([[value_id]], dtype=torch.long, device=device)

        # Input: [dummy, VALUE]
        eval_input = torch.cat([dummy_token, value_token], dim=1)  # [1, 2]

        # Position IDs:
        # - Position 0 (dummy): n - 1 (last prefix position)
        # - Position 1 (VALUE): n (next position after prefix)
        position_ids = torch.tensor([[n - 1, n]], dtype=torch.long, device=device)

        # Attention mask: [1, 1, 2, n + 2]
        # Row 0 (dummy): attend to all prefix (positions 0:n)
        # Row 1 (VALUE): attend to all prefix + dummy
        query_len = 2
        key_len = n + query_len  # prefix (n) + current (2)

        attention_mask = torch.zeros(1, query_len, key_len, device=device)

        # Row 0 (dummy): attend to all prefix
        attention_mask[0, 0, :n] = 1.0

        # Row 1 (VALUE): attend to all prefix + dummy
        attention_mask[0, 1, :n] = 1.0  # All prefix
        attention_mask[0, 1, n] = 1.0   # The dummy token
        # Note: VALUE should NOT attend to itself (causal)

        # Convert to log-space
        attention_mask = torch.where(attention_mask == 1.0, 0.0, float('-inf'))
        attention_mask = attention_mask.unsqueeze(1)  # [1, 1, 2, n+2]

        print(f"Eval input: {eval_input.tolist()}")
        print(f"Position IDs: {position_ids.tolist()}")
        print(f"Attention mask shape: {attention_mask.shape}")

        # Forward with cache
        eval_outputs = base_model(
            eval_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix_kv,
            use_cache=False,
            output_hidden_states=True
        )

        eval_hidden = eval_outputs.hidden_states[-1]  # [1, 2, hidden_dim]

        print(f"Eval hidden shape: {eval_hidden.shape}")

        # Position 0 is dummy (represents prefix last)
        # Position 1 is VALUE
        value_hidden_cached = eval_hidden[:, 1, :]  # [1, hidden_dim]

        print(f"VALUE hidden (cached): shape={value_hidden_cached.shape}")
        print(f"  First 5 dims: {value_hidden_cached[0, :5].tolist()}")
        print(f"  Norm: {value_hidden_cached.norm().item():.6f}")

        # Value head
        value_cached = value_head(value_hidden_cached)
        print(f"VALUE (cached): {value_cached.item():.6f}")

    # ================================================================
    # Comparison
    # ================================================================
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)

    hidden_diff = (value_hidden_direct - value_hidden_cached).abs()
    print(f"Hidden state difference:")
    print(f"  Max: {hidden_diff.max().item():.6f}")
    print(f"  Mean: {hidden_diff.mean().item():.6f}")
    print(f"  Relative: {(hidden_diff.max() / value_hidden_direct.abs().max()).item():.6f}")

    value_diff = abs(value_direct.item() - value_cached.item())
    print(f"\nValue difference: {value_diff:.6f}")
    print(f"  Direct: {value_direct.item():.6f}")
    print(f"  Cached: {value_cached.item():.6f}")

    # ================================================================
    # Method 3: Hybrid - use real prefix last token instead of dummy
    # ================================================================
    print("\n" + "="*60)
    print("Method 3: Cache with real prefix last token (not dummy)")
    print("="*60)

    with torch.no_grad():
        # Use the actual last token from prefix instead of dummy=0
        real_last_token = prefix_ids[:, -1:]  # The actual last token from prefix

        # Input: [real_last_token, VALUE]
        eval_input_v2 = torch.cat([real_last_token, value_token], dim=1)

        print(f"Real last token: {real_last_token.item()}")
        print(f"Eval input v2: {eval_input_v2.tolist()}")

        # Same attention mask and position IDs
        eval_outputs_v2 = base_model(
            eval_input_v2,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix_kv,
            use_cache=False,
            output_hidden_states=True
        )

        eval_hidden_v2 = eval_outputs_v2.hidden_states[-1]
        value_hidden_v2 = eval_hidden_v2[:, 1, :]

        print(f"VALUE hidden (real token): shape={value_hidden_v2.shape}")
        print(f"  First 5 dims: {value_hidden_v2[0, :5].tolist()}")
        print(f"  Norm: {value_hidden_v2.norm().item():.6f}")

        value_v2 = value_head(value_hidden_v2)
        print(f"VALUE (real token): {value_v2.item():.6f}")

        hidden_diff_v2 = (value_hidden_direct - value_hidden_v2).abs()
        print(f"\nHidden difference from direct:")
        print(f"  Max: {hidden_diff_v2.max().item():.6f}")
        print(f"  Mean: {hidden_diff_v2.mean().item():.6f}")

    # ================================================================
    # Method 4: No dummy, just VALUE token directly
    # ================================================================
    print("\n" + "="*60)
    print("Method 4: Direct VALUE with cache (no dummy)")
    print("="*60)

    with torch.no_grad():
        # Just VALUE token
        eval_input_v3 = value_token  # [1, 1]

        # Position ID: n (next after prefix)
        position_ids_v3 = torch.tensor([[n]], dtype=torch.long, device=device)

        # Attention mask: [1, 1, 1, n+1]
        # VALUE attends to all prefix
        attention_mask_v3 = torch.zeros(1, 1, n + 1, device=device)
        attention_mask_v3[0, 0, :n] = 1.0  # Attend to all prefix
        # VALUE does NOT attend to itself (causal)
        attention_mask_v3 = torch.where(attention_mask_v3 == 1.0, 0.0, float('-inf'))
        attention_mask_v3 = attention_mask_v3.unsqueeze(1)  # [1, 1, 1, n+1]

        print(f"Eval input v3: {eval_input_v3.tolist()}")
        print(f"Position IDs: {position_ids_v3.tolist()}")
        print(f"Attention mask shape: {attention_mask_v3.shape}")

        eval_outputs_v3 = base_model(
            eval_input_v3,
            attention_mask=attention_mask_v3,
            position_ids=position_ids_v3,
            past_key_values=prefix_kv,
            use_cache=False,
            output_hidden_states=True
        )

        eval_hidden_v3 = eval_outputs_v3.hidden_states[-1]
        value_hidden_v3 = eval_hidden_v3[:, 0, :]  # Only one position

        print(f"VALUE hidden (no dummy): shape={value_hidden_v3.shape}")
        print(f"  First 5 dims: {value_hidden_v3[0, :5].tolist()}")
        print(f"  Norm: {value_hidden_v3.norm().item():.6f}")

        value_v3 = value_head(value_hidden_v3)
        print(f"VALUE (no dummy): {value_v3.item():.6f}")

        hidden_diff_v3 = (value_hidden_direct - value_hidden_v3).abs()
        print(f"\nHidden difference from direct:")
        print(f"  Max: {hidden_diff_v3.max().item():.6f}")
        print(f"  Mean: {hidden_diff_v3.mean().item():.6f}")

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Direct (EvaluationLM):     {value_direct.item():+.6f}")
    print(f"Cached (dummy token):      {value_cached.item():+.6f} (diff: {abs(value_direct.item() - value_cached.item()):.6f})")
    print(f"Cached (real last token):  {value_v2.item():+.6f} (diff: {abs(value_direct.item() - value_v2.item()):.6f})")
    print(f"Cached (no dummy):         {value_v3.item():+.6f} (diff: {abs(value_direct.item() - value_v3.item()):.6f})")


if __name__ == "__main__":
    test_value_inference_comparison()
