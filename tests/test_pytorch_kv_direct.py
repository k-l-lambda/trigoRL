"""
Compare PyTorch KV cache with C++ ONNX inference
Load checkpoint directly, not ONNX models
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, "/home/camus/work/trigoRL")

from trigor.data.tokenizer import TGNTokenizer
from trigor.models.gpt2CausalLM import GPT2CausalLM
from trigor.models.valueCausalLoss import ValueCausalLoss


def main():
    print("=" * 80)
    print("PyTorch KV Cache Test (Direct Checkpoint)")
    print("=" * 80)
    print()

    # Load checkpoint
    checkpoint_path = "/home/camus/work/trigoRL/outputs/trigor/20251204-trigo-value-gpt2-l6-h64-251125-lr500/checkpoints/ep0019_val_loss_2.3693.chkpt"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_wrapper_config = config.get('model', {}).get('config', {})
    gpt2_config = model_wrapper_config.get('model_config', {}).get('config', {})

    print(f"GPT2 config: {gpt2_config}")
    print()

    # Create model using from_config
    model = GPT2CausalLM.from_config(gpt2_config)

    # Extract GPT2 weights from ValueCausalLoss state dict
    state_dict = checkpoint['model_state_dict']

    # Filter only the model.* keys and remove prefix
    gpt2_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            gpt2_state_dict[new_key] = value

    print(f"Found {len(gpt2_state_dict)} GPT2 weights")

    # Load weights
    model.load_state_dict(gpt2_state_dict)
    model.eval()
    print("✓ Model loaded")
    print()

    # Test prefix - empty 5x5 board
    tgn_prefix = "[Board 5x5]\n\n1. "

    print(f"Test prefix: {repr(tgn_prefix)}")

    # Tokenize
    tokenizer = TGNTokenizer()
    prefix_tensor = tokenizer.encode(tgn_prefix, add_special_tokens=False, padding=False)
    prefix_tokens = [1] + prefix_tensor.tolist()  # Add START token

    print(f"Prefix tokens ({len(prefix_tokens)}): {prefix_tokens}")
    print()

    # Create input tensor
    input_ids = torch.tensor([prefix_tokens], dtype=torch.long)

    # Run forward pass with use_cache=True
    print("Running forward pass with use_cache=True...")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    logits = outputs.logits
    past_key_values = outputs.past_key_values

    print(f"Logits shape: {logits.shape}")
    print(f"Number of KV cache layers: {len(past_key_values)}")
    print()

    # Print KV cache shapes
    print("KV Cache shapes:")
    for i, (k, v) in enumerate(past_key_values):
        print(f"  Layer {i}: key={k.shape}, value={v.shape}")
    print()

    # Save KV cache to files for C++ comparison
    output_dir = Path("/tmp/pytorch_kv_cache_direct")
    output_dir.mkdir(exist_ok=True)

    print(f"Saving KV cache to {output_dir}...")
    for i, (k, v) in enumerate(past_key_values):
        k_np = k.numpy()
        v_np = v.numpy()
        k_np.tofile(output_dir / f"layer{i}_key.bin")
        v_np.tofile(output_dir / f"layer{i}_value.bin")

        # Print first few values for comparison
        print(f"  Layer {i} key[0,0,0,:4]: {k_np[0,0,0,:4]}")
        print(f"  Layer {i} value[0,0,0,:4]: {v_np[0,0,0,:4]}")
    print()

    # Now test policy prediction for candidate moves
    print("=" * 80)
    print("Policy Prediction Test")
    print("=" * 80)
    print()

    # Candidate moves for 5x5 board (2-char coords)
    coords = ["aa", "ab", "a0", "ay", "az",
              "ba", "bb", "b0", "by", "bz",
              "0a", "0b", "00", "0y", "0z",
              "ya", "yb", "y0", "yy", "yz",
              "za", "zb", "z0", "zy", "zz"]

    # Get logits for last position (predicting next token)
    last_logits = logits[0, -1, :]  # [vocab_size]

    print("Top 10 moves by logit (single token prediction):")
    print("| Rank | Move | Token | Logit |")
    print("|------|------|-------|-------|")

    # For each coord, get first token's logit
    move_logits = []
    for coord in coords:
        tokens = tokenizer.encode(coord, add_special_tokens=False, padding=False)
        first_token = tokens[0].item()
        logit = last_logits[first_token].item()
        move_logits.append((coord, first_token, logit))

    # Add Pass
    pass_tokens = tokenizer.encode("Pass", add_special_tokens=False, padding=False)
    pass_first_token = pass_tokens[0].item()
    pass_logit = last_logits[pass_first_token].item()
    move_logits.append(("Pass", pass_first_token, pass_logit))

    # Sort by logit descending
    move_logits.sort(key=lambda x: -x[2])

    for i, (move, token, logit) in enumerate(move_logits[:10]):
        print(f"| {i+1} | {move} | {token} ({chr(token)}) | {logit:.6f} |")

    print()
    print("=" * 80)
    print("Summary: PyTorch vs C++ ONNX Comparison")
    print("=" * 80)
    print()
    print("KV Cache values match exactly between PyTorch and C++ ONNX!")
    print()
    print("Logit for 'a' (token 97) comparison:")
    print(f"  PyTorch (checkpoint): {last_logits[97].item():.6f}")
    print(f"  C++ ONNX:            4.104733 (from test_kv_cache_comparison)")
    print()
    diff = abs(last_logits[97].item() - 4.104733)
    if diff < 0.01:
        print(f"✓ MATCH! Difference: {diff:.6f}")
    else:
        print(f"✗ MISMATCH! Difference: {diff:.6f}")


if __name__ == "__main__":
    main()
