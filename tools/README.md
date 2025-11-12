# TGNDataset CLI Tool

Command-line tool for viewing, exploring, and validating TGNDataset contents. This tool helps verify that the TGNDataset implementation is working correctly and provides insights into the training data.


## Overview

The `view_dataset.py` CLI tool provides multiple functionalities:

- **Statistics**: View dataset-wide statistics (file counts, sizes, vocab info)
- **List Samples**: Browse all available samples in the dataset
- **View Samples**: Inspect individual samples with tokenization details
- **Validate**: Run automated validation checks on the dataset
- **Token Analysis**: Examine tokenization, encoding, and decoding
- **Batch Visualization**: Interactive matplotlib visualization of batches (NEW!)


## Installation

The tool is part of the TrigoRL project and requires the Python environment:

```bash
# Activate the environment
source /home/camus/work/trigoRL/env/bin/activate

# The tool is ready to use
python tools/view_dataset.py --help
```


## Basic Usage

### View Dataset Statistics

Display comprehensive statistics about the dataset:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --stats
```

**Output:**
- Total number of files
- Total and average file sizes
- Vocabulary size
- Maximum sequence length
- Data directory path

### List All Samples

Browse all samples in the dataset:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --list
```

Shows the first 20 samples with their filenames and sizes.

### View a Specific Sample

Inspect a single sample in detail:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0
```

**Output:**
- File information (name, path, size)
- Tensor shapes (input_ids, labels, attention_mask)
- Token statistics (non-padding vs padding tokens)
- Original TGN text content

### Validate Dataset

Run validation checks to ensure the dataset is correctly implemented:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --validate
```

**Validation checks:**
- ✓ Correct tensor types (all torch.Tensor)
- ✓ Consistent shapes (input_ids, labels, attention_mask match)
- ✓ Valid token ranges (0-258)
- ✓ Valid attention masks (binary values: 0 or 1)
- ✓ Proper sequence structure (START token at beginning)

### Interactive Batch Visualization (NEW!)

Visualize batches with comprehensive matplotlib plots:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 4
```

**Features:**
- **Input Token IDs Heatmap**: Visualize token sequences as heatmaps
- **Label Token IDs Heatmap**: See next-token prediction targets
- **Attention Mask**: Binary mask showing valid vs padding tokens
- **Token Distribution**: Top 20 most common tokens in the batch
- **Sequence Statistics**: Non-padding/padding counts, special token counts
- **Per-Sample Distribution**: Stacked bar chart showing token composition

**Controls:**
- Close the window to view the next batch
- Press Ctrl+C to exit the visualization loop

**Visualization includes:**
- Batch size and sequence length information
- File names and sizes for samples in the batch
- Token statistics (mean, min, max for non-padding and padding)
- Special token counts (START, END, PAD)


## Advanced Usage

### View Tokenization Details

Show the actual token IDs for a sample:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --tokens
```

**Output includes:**
- Input token IDs (first 50)
- Label token IDs (first 50)
- Attention mask values
- Special token detection (START, END, PAD)

### Verify Encoding/Decoding

Check that tokenization round-trips correctly:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --decoded --tokens
```

Displays the decoded text from tokens alongside the original text for comparison.

### Customize Token Display

Control how many tokens are shown:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --tokens --max-tokens 100
```

### Hide Original Text

View only tokenization info without the original TGN text:

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --tokens --no-text
```

### Extended Validation

Validate more samples (default is 5):

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --validate --validate-samples 20
```

### Customize Batch Visualization

Control batch size and starting position:

```bash
# Larger batches
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 8

# Start from specific batch
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --start-batch 5

# Combine options
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 2 --start-batch 10
```


## Example Workflows

### Quick Dataset Check

Verify a new dataset is working:

```bash
# Check statistics
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --stats

# Validate implementation
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --validate

# View a few samples
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 1
```

### Debug Tokenization Issues

Investigate tokenization problems:

```bash
# View tokens and compare with decoded text
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --tokens --decoded

# Check multiple samples
for i in {0..5}; do
  python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample $i --tokens
done
```

### Explore Dataset Contents

Browse and understand the training data:

```bash
# List all files
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --list

# View random samples
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 10
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 50
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 99
```

### Visualize Training Batches (NEW!)

Understand batch composition before training:

```bash
# Visualize first few batches
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 4

# Check different batch sizes
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 8

# Start from middle of dataset
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --start-batch 10
```

**What you'll see:**
- Token ID heatmaps showing the distribution across sequences
- Attention masks revealing padding patterns
- Token frequency distribution within the batch
- Sequence length variations across samples
- File names and metadata for each sample


## Configuration Files

The tool works with any training configuration file that contains a `data` section with TGNDataset parameters:

```yaml
# Example config structure
data:
  type: TGNDataset
  data_dir: ${paths.root}/third_party/trigo/trigo-web/tools/output
  max_length: 2048
  min_length: 10
  max_file_size: 10000
  tokenizer_config: {}
```

**Supported configs:**
- `configs/training/trigo-gpt2.yaml` - GPT-2 training config
- `configs/training/trigo-llama.yaml` - LLaMA training config
- `configs/training/trigo-rwkv.yaml` - RWKV training config
- `configs/training/trigo-xlstm.yaml` - xLSTM training config


## CLI Options Reference

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `config` | Path to training config YAML file |

### Optional Arguments

| Option | Description |
|--------|-------------|
| `--stats` | Display dataset statistics |
| `--list` | List all samples in the dataset |
| `--sample IDX` | Display specific sample by index |
| `--validate` | Run validation checks |
| `--validate-samples N` | Number of samples to validate (default: 5) |
| `--visualize` | Interactive batch visualization with matplotlib |
| `--batch-size N` | Batch size for visualization (default: 4) |
| `--start-batch N` | Starting batch index for visualization (default: 0) |
| `--tokens` | Show token sequences when displaying samples |
| `--decoded` | Show decoded text from tokens |
| `--no-text` | Hide original text when displaying samples |
| `--max-tokens N` | Maximum number of tokens to display (default: 50) |


## Understanding the Output

### Token Statistics

```
Token Statistics:
  Non-padding tokens: 304
  Padding tokens:     1743
  Sequence length:    2047
```

- **Non-padding tokens**: Actual content from the TGN file + special tokens
- **Padding tokens**: PAD tokens (256) added to reach max_length
- **Sequence length**: Always max_length - 1 (due to input/label shift)

### Special Tokens

The TGNByteTokenizer uses three special tokens:

- **START (257)**: Added at the beginning of input sequences
- **END (258)**: Added at the end of sequences
- **PAD (256)**: Used to pad sequences to max_length

### Token ID Ranges

- **0-255**: Standard UTF-8 byte values (actual TGN text content)
- **256**: PAD token
- **257**: START token
- **258**: END token
- **Vocabulary size**: 259 total tokens

### Input/Label Relationship

For next-token prediction:

```
Input:  [START, tok1, tok2, tok3, ..., tokN-1]
Label:  [tok1, tok2, tok3, ..., tokN-1, END]
```

This shift ensures the model learns to predict the next token at each position.


## Troubleshooting

### No .tgn files found

**Error**: `ValueError: No .tgn files found in <directory>`

**Solution**: Ensure the `data_dir` in your config points to a directory containing .tgn files. For TrigoRL, this should be:
```yaml
data_dir: ${paths.root}/third_party/trigo/trigo-web/tools/output
```

Generate TGN files if needed:
```bash
cd third_party/trigo/trigo-web
npm run generate:games
```

### Sample index out of range

**Error**: `Sample index X out of range [0, Y]`

**Solution**: Check the dataset size first with `--stats` or `--list`, then use a valid index.

### Import errors

**Error**: `ModuleNotFoundError: No module named 'trigor'`

**Solution**: Make sure you're running from the project root and the environment is activated:
```bash
cd /home/camus/work/trigoRL
source env/bin/activate
python tools/view_dataset.py ...
```


## Implementation Notes

### Dataset Validation

The validation checks ensure:

1. **Tensor Types**: All outputs are torch.Tensor
2. **Shape Consistency**: input_ids, labels, and attention_mask have matching shapes
3. **Token Range**: All token IDs are in the valid range [0, 258]
4. **Attention Mask**: Only contains 0 (padding) or 1 (valid token)
5. **Sequence Structure**: Input starts with START token (257)

### Performance

The tool is designed for development and debugging, not for processing large datasets. For batch operations, consider using the dataset directly in Python:

```python
from omegaconf import OmegaConf
from trigor.data import TGNDataset

cfg = OmegaConf.load('configs/training/trigo-gpt2.yaml')
dataset = TGNDataset.from_config(cfg.data)

# Process in batch
for i in range(len(dataset)):
    sample = dataset[i]
    # Your processing here
```


## Related Documentation

- [TGNDataset Implementation](../trigor/data/tgn_dataset.py) - Dataset class
- [TGNByteTokenizer](../trigor/data/tokenizer.py) - Tokenizer implementation
- [Training Configs](../configs/training/README.md) - Configuration documentation
- [TGN Format Specification](../third_party/trigo/docs/tgn-format-spec.md) - TGN notation


## Examples

### Example 1: Quick Validation

```bash
# Check if dataset is working correctly
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --validate
```

**Expected output:**
```
✓ Validation PASSED!
  All 5 samples validated successfully
  ✓ Correct tensor types
  ✓ Consistent shapes
  ✓ Valid token ranges
  ✓ Valid attention masks
  ✓ Proper sequence structure
```

### Example 2: Inspect Tokenization

```bash
# View a sample with full tokenization details
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --tokens --decoded
```

**Expected output:**
- Original TGN text
- Token IDs for input and labels
- Attention mask values
- Decoded text from tokens (should match original)
- Special token indicators

### Example 3: Dataset Overview

```bash
# Get complete overview
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --stats --list
```

**Expected output:**
- Statistics (file count, sizes, vocab info)
- List of first 20 samples with filenames and sizes


## Development

### Adding New Features

The tool is structured into modular functions:

- `load_dataset_from_config()` - Load dataset from config
- `display_dataset_stats()` - Show statistics
- `display_sample()` - Show individual sample details
- `validate_dataset()` - Run validation checks
- `list_samples()` - List all samples

To add new functionality, create a new function and add a CLI argument in `main()`.

### Testing

Test the tool with all available configs:

```bash
for config in configs/training/trigo-*.yaml; do
  echo "Testing $config"
  python tools/view_dataset.py "$config" --validate
done
```


## Summary

The TGNDataset CLI tool provides a comprehensive way to:
- ✓ Verify dataset implementation correctness
- ✓ Understand tokenization and encoding
- ✓ Debug data loading issues
- ✓ Explore training data contents
- ✓ Validate data preprocessing pipeline

Use this tool during development to ensure your dataset is properly configured before starting training runs.
