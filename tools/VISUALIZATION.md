# Batch Visualization Examples

This document demonstrates the batch visualization feature of the TGNDataset CLI tool.

## Overview

The `--visualize` flag enables interactive batch visualization using matplotlib. This provides a comprehensive view of how data is structured in training batches.

## Visualization Components

Each visualization displays 6 key components in a single window:

### 1. Input Token IDs Heatmap (Top Left)
- **Size**: batch_size × sequence_length
- **Colormap**: viridis (yellow=high token IDs, purple=low)
- Shows the actual token IDs fed to the model
- Useful for spotting patterns and padding

### 2. Label Token IDs Heatmap (Middle Left)
- **Size**: batch_size × sequence_length
- **Colormap**: plasma (yellow=high, purple=low)
- Shows the next-token prediction targets
- Labels are shifted by 1 position from inputs

### 3. Attention Mask (Bottom Left)
- **Size**: batch_size × sequence_length
- **Colormap**: RdYlGn (green=1/valid, red=0/padding)
- Binary mask indicating which tokens are valid (1) vs padding (0)
- Clearly shows where sequences end and padding begins

### 4. Top 20 Token Distribution (Top Right)
- Horizontal bar chart showing most frequent tokens in the batch
- Excludes padding tokens (256)
- Helps understand token frequency patterns
- Useful for checking vocabulary usage

### 5. Sequence Statistics (Middle Right)
- Text panel with comprehensive batch statistics:
  - Batch size and sequence length
  - Non-padding token statistics (mean, min, max)
  - Padding token statistics
  - Special token counts (START=257, END=258, PAD=256)

### 6. Token Distribution per Sample (Bottom Right)
- Stacked bar chart for each sample in the batch
- **Coral bars**: Valid tokens (actual content)
- **Gray bars**: Padding tokens
- Shows length variation across samples

### Footer Information
- Lists all samples in the batch with filenames and sizes
- Helps track which TGN files are being processed

## Usage

### Basic Visualization

```bash
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize
```

This will:
1. Load the dataset from the config
2. Create batches of size 4 (default)
3. Display the first batch
4. Wait for you to close the window
5. Show the next batch
6. Repeat until all batches are shown or you press Ctrl+C

### Custom Batch Size

```bash
# Smaller batches for detailed inspection
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 2

# Larger batches for overview
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --batch-size 8
```

### Start from Specific Batch

```bash
# Skip first 5 batches
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --visualize --start-batch 5
```

## Interpreting the Visualization

### What to Look For

1. **Padding Patterns** (Attention Mask):
   - Green regions show actual content
   - Red regions show padding
   - If all samples have similar padding, sequences are similar length
   - If padding varies greatly, dataset has diverse sequence lengths

2. **Token Distribution**:
   - High frequency of START (257) indicates correct tokenization
   - PAD count should match padding in attention mask
   - Token distribution should reflect TGN notation patterns

3. **Sequence Length Variation**:
   - Per-sample bar chart shows length diversity
   - Consistent lengths = efficient batching
   - High variation = more padding waste

4. **Token ID Patterns** (Heatmaps):
   - Vertical patterns indicate similar content at same positions
   - Sudden color changes show sequence boundaries
   - Padding tokens (256) appear as specific color band

### Example Observations

**Typical TGN Dataset:**
- Most tokens are ASCII characters (32-122): lowercase, digits, punctuation
- START token (257) always at position 0 in input_ids
- END token (258) appears in labels where sequences end
- PAD token (256) fills remaining positions
- Average sequence length: 300-1500 tokens (for game notation)
- Padding: 30-85% of sequence length (depends on file size)

## Test Results

From our test run with trigo-gpt2.yaml:

```
Dataset: 100 TGN files
Batch size: 4
Sequence length: 2047 (max_length - 1)

Batch 1 Statistics:
  Non-padding tokens: Mean 1161.5, Min 304, Max 2045
  Padding tokens: Mean 885.5, Min 2, Max 1743
  START tokens: 4 (one per sample)
  END tokens: 4 (one per sample)
  PAD tokens: 3542 (varies by sample)
```

## Technical Details

### Color Maps
- **viridis**: Good for sequential data, colorblind-friendly
- **plasma**: Similar to viridis but with different emphasis
- **RdYlGn**: Red-Yellow-Green, perfect for binary masks

### Figure Size
- 16×10 inches (1600×1000 pixels at 100 DPI)
- High resolution: 2050×1489 pixels at 150 DPI when saved
- Optimized for readability on modern displays

### Performance
- Visualization generation: ~0.5 seconds per batch
- Memory usage: ~200MB for matplotlib window
- Non-interactive mode (save to file): ~0.3 seconds per batch

## Saving Visualizations

To save visualizations to files instead of displaying interactively, you can modify the test script:

```python
# See tests/test_visualization.py for example
# Uses matplotlib's 'Agg' backend to save PNG files
```

## Use Cases

1. **Dataset Validation**: Verify tokenization and batching work correctly
2. **Training Preparation**: Understand batch composition before training
3. **Debugging**: Identify issues with padding, token ranges, or masks
4. **Documentation**: Generate visualizations for papers or reports
5. **Hyperparameter Tuning**: Decide optimal batch size based on padding waste

## Limitations

- Interactive mode requires GUI display (X11/Wayland on Linux)
- Large batch sizes (>16) may make heatmaps hard to read
- Very long sequences compressed in horizontal axis
- Color perception may vary depending on display calibration

## Troubleshooting

**Issue**: "cannot connect to X server"
**Solution**: Use the test script with 'Agg' backend, or enable X11 forwarding

**Issue**: Window is too small/large
**Solution**: Resize manually or modify `figsize` in the code

**Issue**: Visualization is slow
**Solution**: Use smaller batch sizes or save to file instead of interactive mode

## Future Enhancements

Potential improvements for the visualization:
- [ ] Zoom into specific regions of heatmaps
- [ ] Compare multiple batches side-by-side
- [ ] Add token ID to character mapping overlay
- [ ] Export statistics to CSV
- [ ] Animation mode to auto-advance batches
- [ ] Customizable color schemes

## Conclusion

The batch visualization feature provides comprehensive insights into how the TGNDataset structures data for training. It's an essential tool for understanding your data pipeline and ensuring everything is working correctly before starting expensive training runs.
