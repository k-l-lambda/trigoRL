# TGN Dataset Loader Usage Example

This document demonstrates how to use the TGN byte-level dataset loader.

## Quick Start

```python
from trigorl.data import TGNByteTokenizer, TGNDataset
from torch.utils.data import DataLoader

# 1. Create tokenizer
tokenizer = TGNByteTokenizer()

# 2. Create dataset
dataset = TGNDataset(
    data_dir="third_party/trigo/trigo-web/tools/output",
    tokenizer=tokenizer,
    max_length=2048
)

# 3. Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=TGNDataset.collate_batch,  # Use static method
    num_workers=0
)

# 4. Iterate through batches
for batch in dataloader:
    input_ids = batch['input_ids']          # [B, 2047]
    labels = batch['labels']                # [B, 2047]
    attention_mask = batch['attention_mask'] # [B, 2047]

    # Your training code here
    pass
```

## Tokenizer Details

### Vocabulary
- **Size**: 259 tokens
- **0-255**: Standard UTF-8 bytes
- **256**: PAD token (padding)
- **257**: START token (beginning of sequence)
- **258**: END token (end of sequence)

### Example

```python
tokenizer = TGNByteTokenizer()

# Encode
text = "[Board 5x5x5]\n\n1. 000 y00\n"
tokens = tokenizer.encode(text, max_length=64)
# Output: tensor([257, 91, 66, 111, ..., 258, 256, 256, ...])

# Decode
decoded = tokenizer.decode(tokens)
# Output: "[Board 5x5x5]\n\n1. 000 y00\n"
```

## Dataset Details

### What it provides

Each item in the dataset contains:
- `input_ids`: Token sequence for input [max_length-1]
- `labels`: Token sequence for targets [max_length-1]
- `attention_mask`: Binary mask (1=valid, 0=padding) [max_length-1]

### Next-token prediction format

```
Original tokens: [START, tok1, tok2, tok3, ..., tokN, END, PAD, PAD, ...]
Input:           [START, tok1, tok2, tok3, ..., tokN-1]
Labels:          [tok1, tok2, tok3, ..., tokN-1, END]
```

## Dataset Statistics

From the test output:
```
Number of files: 100
Total bytes: 116,158
Average bytes per game: 1,162
Min bytes: 39
Max bytes: 4,597
```

## Advanced Usage

### Filtering datasets

```python
# Filter by file size
dataset = TGNDataset(
    data_dir="path/to/tgn/files",
    tokenizer=tokenizer,
    min_length=100,      # Min 100 bytes
    max_file_size=3000,  # Max 3KB
)

# Custom filter function
def my_filter(file_path):
    # Only include files with specific pattern
    return "game_" in file_path.name

dataset = TGNDataset(
    data_dir="path/to/tgn/files",
    tokenizer=tokenizer,
    filter_fn=my_filter
)
```

### Getting dataset statistics

```python
stats = dataset.get_stats()
print(f"Number of files: {stats['num_files']}")
print(f"Average size: {stats['avg_bytes']:.2f} bytes")
print(f"Vocabulary size: {stats['vocab_size']}")
```

### Accessing individual files

```python
# Get raw text
text = dataset.get_text(idx=0)

# Get file info
info = dataset.get_file_info(idx=0)
print(info['name'])        # Filename
print(info['size_bytes'])  # File size
print(info['path'])        # Full path
```

## Training Example

```python
import torch
import torch.nn as nn
from torch.optim import Adam

from trigorl.data import TGNByteTokenizer, TGNDataset
from trigorl.models import PolicyValueNetwork  # Your model

# Setup
tokenizer = TGNByteTokenizer()
dataset = TGNDataset("path/to/tgn/files", tokenizer, max_length=1024)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model (example)
model = PolicyValueNetwork(
    observation_dim=tokenizer.VOCAB_SIZE,
    action_dim=tokenizer.VOCAB_SIZE,
    hidden_dims=[512, 512]
)
optimizer = Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_TOKEN_ID)

# Training loop
model.train()
for epoch in range(10):
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']

        # Forward pass
        logits = model(input_ids)

        # Compute loss (only on valid tokens)
        loss = criterion(
            logits.view(-1, tokenizer.VOCAB_SIZE),
            labels.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## File Structure

```
trigorl/data/
├── __init__.py           # Package exports
├── tokenizer.py          # TGNByteTokenizer class
└── tgn_dataset.py        # TGNDataset class
```

## Testing

Run the test script:
```bash
python test_tgn_dataset.py
```

Expected output:
```
================================================================================
TGN DATASET LOADER TEST SUITE
================================================================================
...
✓ ALL TESTS PASSED!
================================================================================
```

## Notes

- The tokenizer is **stateless** and thread-safe
- The dataset **caches file paths** on initialization for fast loading
- **Context window**: 2048 bytes covers ~95% of games
- **Byte-level encoding**: No vocabulary size issues, works with any notation
- **Special tokens** use byte values above 255 to avoid conflicts

## Future Enhancements

Potential additions (not implemented):
- Data augmentation (rotation, reflection, color swap)
- Board state extraction from TGN notation
- Reward computation from game outcomes
- Multi-file dataset support (split large datasets)
- Dynamic padding (per-batch instead of max_length)
