
> This is a reinforcement learning lab project. Based on the game of Trigo, a board game of Go in 3D space.
> Technical basic: pyTorch + transformers, wanDB for training stats, onnx exporting for model weights support.


## 2025/11/11


> Plan the framework for this project, investigate the configuration system in `../deep-starry` project.
> Set up modernized deep learning training basic modules based on a similar YAML config system to deep-starry.

<details>
<summary>Complete RL training framework implemented</summary>

Investigated deep-starry's configuration system and implemented a modernized version for TrigoRL:

**Configuration System**:
- Hydra + OmegaConf for composable YAML configs (modernization from deep-starry's pure PyYAML)
- Hierarchical config structure: env + agent + training
- CLI overrides supported
- Auto-generated experiment directories

**Core Infrastructure**:
- Agent registry with factory pattern (inspired by deep-starry)
- Environment registry for flexible env creation
- WandbLogger for experiment tracking (modernization from TensorBoard)
- CheckpointManager with best model tracking

**Training System**:
- Custom RLTrainer adapted from deep-starry's Trainer
- Episode-based training loop (vs epoch-based)
- Evaluation episodes with metric monitoring
- Checkpoint save/load with state persistence

**Initial Components**:
- RandomAgent for testing pipeline
- DummyEnv for framework validation
- MLP and PolicyValueNetwork architectures

**Project Structure**:
```
trigoRL/
├── configs/              # Hydra configuration files
├── trigor/             # Main package
│   ├── agents/          # Agent registry and implementations
│   ├── envs/            # Environment registry and wrappers
│   ├── training/        # Custom RL trainer
│   ├── models/          # Neural network architectures
│   └── utils/           # Logger, checkpoint manager
├── train.py             # Main entry point
└── requirements.txt     # Dependencies
```

Next steps: Test the framework end-to-end with random agent + dummy environment.
</details>


> Investigate transformers backbone model layers suitable for RL research. Requirements: simple implementation (no complex optimized kernels), comparable architectures, coverage of MHA, GQA, linear attention (with/without TTT), and LSTM.

<details>
<summary>Comprehensive investigation of Hugging Face transformers library completed</summary>

### Executive Summary

Investigated the Hugging Face transformers library (393 model implementations) and identified clean, vanilla implementations of attention mechanisms suitable for RL research. Found excellent coverage of MHA, GQA, Linear Attention (RWKV), and LSTM (xLSTM) - all with pure PyTorch implementations and no complex optimized kernels.

### Key Findings

**✅ Found Implementations:**

| Attention Type | Best Implementation | File Size | Code Quality | Recommendation |
|----------------|-------------------|-----------|--------------|----------------|
| **MHA** | GPT-2 | 1302 lines | ⭐⭐⭐⭐⭐ | Most educational |
| **GQA** | LLaMA | 532 lines | ⭐⭐⭐⭐⭐ | **PRIMARY CHOICE** |
| **Linear Attention** | RWKV | 780 lines | ⭐⭐⭐⭐ | Excellent |
| **LSTM** | xLSTM | 1630 lines | ⭐⭐⭐ | Modern variant |
| **TTT** | ❌ Not found | - | - | Needs custom implementation |

### 1. Multi-Head Attention (MHA) - GPT-2

**Location:** `third_party/transformers/src/transformers/models/gpt2/modeling_gpt2.py`

**Key Components:**
- Lines 53-89: `eager_attention_forward()` - Pure PyTorch attention (no Flash Attention)
- Lines 92-270: `GPT2Attention` class - Standard multi-head attention

**Advantages:**
- Simplest, most educational implementation
- No external dependencies or optimized kernels
- Easy to modify for experiments
- Clean causal masking implementation

**Configuration:**
```python
GPT2Config(
    n_embd=768,          # Hidden size
    n_head=12,           # Number of attention heads
    n_layer=12,          # Number of layers
    attn_pdrop=0.1,      # Attention dropout
)
```

### 2. Grouped Query Attention (GQA) - LLaMA ⭐ **RECOMMENDED**

**Location:** `third_party/transformers/src/transformers/models/llama/modeling_llama.py`

**Key Components:**
- Lines 188-197: `repeat_kv()` - Core GQA function that expands K/V heads
- Lines 226-293: `LlamaAttention` - Main attention class
- Lines 138-169: `apply_rotary_pos_emb()` - RoPE positional encoding

**Key Innovation - repeat_kv:**
```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands key/value heads to match query heads.
    From (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

**Flexible Configuration:**
```python
# Multi-Head Attention (MHA) - all heads independent
config = LlamaConfig(num_attention_heads=32, num_key_value_heads=32)

# Grouped Query Attention (GQA) - 4 groups
config = LlamaConfig(num_attention_heads=32, num_key_value_heads=8)

# Multi-Query Attention (MQA) - single K/V
config = LlamaConfig(num_attention_heads=32, num_key_value_heads=1)
```

**Advantages:**
- **Cleanest code** (only 532 lines - smallest model file)
- **Most flexible** (supports MHA/GQA/MQA via config)
- Production-quality, battle-tested
- **Best for architecture comparison experiments**
- Pure PyTorch, no external dependencies

### 3. Linear Attention - RWKV

**Location:** `third_party/transformers/src/transformers/models/rwkv/modeling_rwkv.py`

**Key Components:**
- Lines 163-204: `rwkv_linear_attention_cpu()` - Pure PyTorch implementation
- Lines 218-285: `RwkvSelfAttention` - Main attention class

**Key Features:**
- **No softmax:** Uses exponential time decay instead
- **Linear complexity:** O(N·D²) vs O(N²·D) for standard attention
- **Recurrent state:** Can process sequences autoregressively with constant memory
- **Time mixing:** Interpolates current and previous timesteps

**Complexity Comparison:**

| Feature | Standard Attention | RWKV |
|---------|-------------------|------|
| Time Complexity | O(N² · D) | O(N · D²) |
| Memory | O(N²) | O(D²) |
| Parallelization | Full sequence | Sequential (training) |
| Long sequences | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Advantages:**
- Linear complexity for long sequences
- Constant memory inference
- Pure PyTorch CPU implementation available
- No softmax (uses WKV mechanism)

### 4. Extended LSTM (xLSTM)

**Location:** `third_party/transformers/src/transformers/models/xlstm/modeling_xlstm.py`

**Key Components:**
- Lines 67-155: `mlstm_chunkwise_recurrent_fw_C()` - Recurrent state computation
- Lines 315-378: `mlstm_chunkwise_native_autograd()` - Main mLSTM function

**Key Innovation - Matrix-valued States:**

Unlike standard LSTM (scalar cell state), xLSTM uses **matrix-valued cell states**:

| Feature | Standard LSTM | mLSTM (xLSTM) |
|---------|---------------|---------------|
| Cell state | Scalar | **Matrix** |
| Gating | Sigmoid | **Exponential (log-space)** |
| Parallelization | Sequential | **Chunk-wise** |
| Normalization | Optional | **Built-in (vecN)** |

**Advantages:**
- Modern LSTM variant with better performance
- Chunk-wise parallelization for efficiency
- Pure PyTorch implementation available (no external dependencies required)
- Multi-head architecture

### 5. TTT (Test-Time Training) - NOT FOUND

Searched the entire transformers library - **no TTT attention implementations found**.

**Closest Alternatives:**
1. **RWKV:** Linear attention with recurrent state (similar concept)
2. **xLSTM:** Adaptive recurrent processing
3. **Mamba:** State-space models (requires `mamba_ssm` package)

TTT will need to be implemented separately.

### Architecture Comparison

**Complexity and Efficiency:**

| Attention Type | Time Complexity | Space Complexity | Long Sequences | Parallelization |
|---------------|----------------|------------------|----------------|-----------------|
| **MHA (GPT-2)** | O(N²·D) | O(N²) | ⭐⭐ | Full |
| **GQA (LLaMA)** | O(N²·D) | O(N²) | ⭐⭐ | Full |
| **Linear (RWKV)** | O(N·D²) | O(D²) | ⭐⭐⭐⭐⭐ | Partial |
| **LSTM (xLSTM)** | O(N·D²) | O(D²) | ⭐⭐⭐⭐ | Chunk-wise |

**Code Quality:**

| Model | Lines | Simplicity | Documentation | Pure PyTorch | Research-Friendly |
|-------|-------|------------|---------------|--------------|-------------------|
| **LLaMA** | 532 | ⭐⭐⭐⭐⭐ | Excellent | ✅ | ⭐⭐⭐⭐⭐ |
| **Mistral** | 507 | ⭐⭐⭐⭐⭐ | Excellent | ✅ | ⭐⭐⭐⭐⭐ |
| **GPT-2** | 1302 | ⭐⭐⭐⭐ | Excellent | ✅ | ⭐⭐⭐⭐ |
| **RWKV** | 780 | ⭐⭐⭐⭐ | Good | ✅ | ⭐⭐⭐⭐ |
| **xLSTM** | 1630 | ⭐⭐⭐ | Good | ✅ | ⭐⭐⭐ |

### Avoiding Flash Attention / Optimized Kernels

All models support multiple attention implementations. To use pure PyTorch eager attention:

```python
config._attn_implementation = "eager"  # No Flash Attention or optimized kernels
```

**Evidence in code:**
- GPT-2 Line 246-249: Attention interface selection
- LLaMA Line 276-278: Same pattern
- All models default to "eager" mode

### Implementation Strategy for RL Research

**Phase 1: Start with LLaMA (GQA)**
```
Extract from: third_party/transformers/src/transformers/models/llama/modeling_llama.py
  - Line 53-70:   LlamaRMSNorm
  - Line 138-169: apply_rotary_pos_emb (RoPE)
  - Line 188-197: repeat_kv (GQA core) ⭐
  - Line 226-293: LlamaAttention (main class) ⭐
```

**Reason:** Cleanest code (532 lines), easiest to modify, supports MHA/GQA/MQA switching

**Phase 2: Add GPT-2 as MHA Baseline**
```
Extract from: third_party/transformers/src/transformers/models/gpt2/modeling_gpt2.py
  - Line 53-89:  eager_attention_forward
  - Line 92-270: GPT2Attention
```

**Reason:** Educational reference for standard MHA

**Phase 3: Experiment with RWKV (Linear Attention)**
```
Extract from: third_party/transformers/src/transformers/models/rwkv/modeling_rwkv.py
  - Line 163-204: rwkv_linear_attention_cpu ⭐
  - Line 218-285: RwkvSelfAttention
```

**Reason:** Linear complexity, suitable for long sequences

**Phase 4: (Optional) xLSTM**
```
Extract from: third_party/transformers/src/transformers/models/xlstm/modeling_xlstm.py
  - Line 315-378: mlstm_chunkwise_native_autograd ⭐
```

**Reason:** Recurrent model comparison

### Modular Architecture Design

**Strategy for Easy Swapping:**

```python
class FlexibleDecoderLayer(nn.Module):
    def __init__(self, config, attention_class):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)

        # Inject different attention types
        self.attn = attention_class(config, layer_idx=0)

        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attn(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

# Usage - swap attention by changing class
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.rwkv.modeling_rwkv import RwkvSelfAttention

layer_gqa = FlexibleDecoderLayer(config, LlamaAttention)
layer_linear = FlexibleDecoderLayer(config, RwkvSelfAttention)
```

### Key Advantages Summary

✅ **All implementations are pure PyTorch** - No external dependencies (xLSTM optional)
✅ **No Flash Attention** - Can use `config._attn_implementation = "eager"`
✅ **Modular design** - Easy to swap and compare
✅ **Production-quality** - Well-documented, battle-tested
✅ **Fully controllable** - No complex optimized kernels, easy to debug

### Quick Reference

**File Locations:**

```
third_party/transformers/src/transformers/

MHA (GPT-2):
└── models/gpt2/modeling_gpt2.py (1302 lines)

GQA (LLaMA):
└── models/llama/modeling_llama.py (532 lines) ⭐ PRIMARY

GQA (Mistral):
└── models/mistral/modeling_mistral.py (507 lines)

Linear Attention (RWKV):
└── models/rwkv/modeling_rwkv.py (780 lines)

LSTM (xLSTM):
└── models/xlstm/modeling_xlstm.py (1630 lines)
```

### Recommendations

**Primary Choice:** **LLaMA** - Use as main foundation
- Cleanest code (532 lines)
- Easy to switch between MHA/GQA/MQA via configuration
- Production-quality, well-tested
- Best for architecture comparison experiments

**Baseline:** **GPT-2** - Standard MHA reference
- Most educational
- Simplest implementation
- Good for understanding basic attention

**Experimental:** **RWKV** - Linear attention alternative
- Linear complexity
- Suitable for long sequences
- Interesting alternative to softmax

**Optional:** **xLSTM** - Recurrent baseline
- Modern LSTM variant
- Good for comparison with attention

### Next Steps

1. Extract LLaMA attention module (Lines 188-293)
2. Create unified configuration system for attention type switching
3. Implement flexible decoder layer with swappable attention
4. Test MHA vs GQA vs MQA comparisons
5. Add RWKV linear attention as alternative
6. (Optional) Implement TTT attention separately

</details>


> Write a dataset loader for TGN with byte-based tokenizer (no word participle). Use files in third_party/trigo/trigo-web/tools/output as examples.

<details>
<summary>TGN byte-level dataset loader implemented</summary>

### Implementation Summary

Created a standalone PyTorch dataset loader for TGN (Trigo Game Notation) files with byte-level tokenization for transformer-based sequence modeling.

### Components Implemented

**1. TGNByteTokenizer** (`trigorl/data/tokenizer.py`)
- Vocabulary size: 259 (256 bytes + 3 special tokens)
- Byte-level encoding: UTF-8 text → byte tokens (0-255)
- Special tokens: PAD (256), START (257), END (258)
- Support for batch encoding/decoding
- Max length truncation and padding

**2. TGNDataset** (`trigorl/data/tgn_dataset.py`)
- PyTorch Dataset for loading TGN files
- Next-token prediction format (input_ids → labels)
- Attention masks for valid tokens
- File filtering by size
- Dataset statistics and metadata

**3. Package Exports** (`trigorl/data/__init__.py`)
- Clean API: `TGNByteTokenizer`, `TGNDataset`, `collate_tgn_batch`

### Dataset Statistics

Loaded from `third_party/trigo/trigo-web/tools/output`:
- **100 TGN files**
- **Total size**: 116,158 bytes
- **Average**: 1,162 bytes per game (~101 moves)
- **Range**: 39 to 4,597 bytes
- **Context window**: 2048 bytes (covers ~95% of games)

### Data Format

Each batch item contains:
```python
{
    'input_ids': torch.Tensor,      # [max_length-1] - input sequence
    'labels': torch.Tensor,         # [max_length-1] - target sequence
    'attention_mask': torch.Tensor, # [max_length-1] - valid token mask
}
```

### Key Design Decisions

**Byte-level tokenization advantages:**
- Fixed vocabulary (259 tokens)
- No OOV (out-of-vocabulary) issues
- Works with any board size/coordinate notation
- No preprocessing required
- Compact representation

**No parsing phase:**
- Model learns structure from raw bytes
- Simpler implementation
- Suitable for transformer models
- Can add parsing later if needed

### Testing

Created comprehensive test suite (`test_tgn_dataset.py`):
- ✅ Tokenizer encode/decode roundtrip
- ✅ Dataset loads all 100 files
- ✅ Batch shapes correct
- ✅ DataLoader iteration works
- ✅ Attention masks valid

All tests passed successfully.

### Usage Example

```python
from trigorl.data import TGNByteTokenizer, TGNDataset
from torch.utils.data import DataLoader

# Setup
tokenizer = TGNByteTokenizer()
dataset = TGNDataset(
    data_dir="third_party/trigo/trigo-web/tools/output",
    tokenizer=tokenizer,
    max_length=2048
)

# DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Iterate
for batch in dataloader:
    input_ids = batch['input_ids']          # [8, 2047]
    labels = batch['labels']                # [8, 2047]
    attention_mask = batch['attention_mask'] # [8, 2047]
```

### Documentation

- Usage guide: `docs/tgn_dataset_usage.md`
- Detailed examples and training code
- API reference
- Advanced filtering options

### Next Steps

Potential enhancements (not yet implemented):
- Data augmentation (rotation, reflection, color swap)
- Board state extraction from TGN notation
- Generate larger dataset (10K+ games)
- Train/validation split utilities
- Integration with RL training pipeline

</details>


## 2025/11/12


> Write a dataset factory, allow create dataset from config file.

<details>
<summary>Dataset factory with registry pattern implemented</summary>

### Implementation Summary

Created a config-driven dataset factory following the same registry pattern used for agents and environments, enabling declarative dataset creation through YAML configuration files.

### Components Implemented

**1. Dataset Registry** (`trigor/data/registry.py`)
- Registry pattern for dataset type management
- `DATASETS` dictionary mapping type names to classes
- `register_dataset()` - Register custom dataset types
- `make_dataset()` - Factory function to create datasets from config
- `list_datasets()` - List all registered dataset types
- Special handling for TGNDataset tokenizer initialization

**2. Configuration Files** (`configs/dataset/`)
Three preconfigured dataset variants:
- `tgn_default.yaml` - Standard (max_length=2048, batch_size=8)
- `tgn_small.yaml` - Fast testing (max_length=512, batch_size=16)
- `tgn_large.yaml` - Full sequences (max_length=4096, batch_size=4)

All configs support Hydra interpolation (e.g., `${paths.root}`)

**3. Main Config Integration** (`configs/config.yaml`)
- Added `dataset: tgn_default` to default config hierarchy
- Enables CLI overrides: `python train.py dataset=tgn_small`

**4. Package Exports** (`trigor/data/__init__.py`)
Extended exports with factory functions:
- `make_dataset`, `register_dataset`, `list_datasets`, `DATASETS`

### Architecture

```
trigor/data/
├── __init__.py          # API exports
├── registry.py          # Factory + registry (NEW)
├── tgn_dataset.py       # TGNDataset implementation
└── tokenizer.py         # TGNByteTokenizer

configs/dataset/
├── tgn_default.yaml     # Standard config (NEW)
├── tgn_small.yaml       # Small config (NEW)
└── tgn_large.yaml       # Large config (NEW)
```

### Usage Examples

**From Python code:**
```python
from trigor.data import make_dataset

config = {
    'type': 'TGNDataset',
    'data_dir': 'data/tgn_games',
    'max_length': 512,
}
dataset = make_dataset(dataset_type=config['type'], config=config)
```

**With Hydra config:**
```python
@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    dataset = make_dataset(cfg.dataset.type, cfg.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.dataloader.batch_size,
        collate_fn=TGNDataset.collate_batch
    )
```

**CLI overrides:**
```bash
python train.py dataset=tgn_small
python train.py dataset=tgn_default dataset.max_length=1024
python train.py dataset.data_dir=/custom/path
```

### YAML Config Structure

```yaml
type: TGNDataset
data_dir: ${paths.root}/third_party/trigo/trigo-web/tools/output
max_length: 2048
min_length: 10
max_file_size: 10000
tokenizer_config: {}

dataloader:
  batch_size: 8
  shuffle: true
  num_workers: 4
  pin_memory: true
```

### Key Features

**Config-driven creation:**
- Declarative dataset setup via YAML
- Supports Hydra's composition and overrides
- Reproducible configurations

**Type safety:**
- Factory validates dataset types before instantiation
- Clear error messages for unregistered types

**Extensibility:**
```python
from trigor.data import register_dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, param):
        ...

register_dataset('CustomDataset', CustomDataset)
```

**Consistency:**
- Same pattern as agent/environment registries
- Unified architecture across framework

### Testing

Created comprehensive test suite (`test_dataset_factory.py`):
- ✅ Registry functions (list_datasets)
- ✅ Config-based dataset creation
- ✅ Dataset iteration and batching
- ✅ DataLoader integration
- ✅ YAML config loading
- ✅ All 5 test sections passed

**Test results:**
```
Registered datasets: ['TGNDataset']
✓ Created dataset with 100 files
✓ Tensor shapes consistent
✓ Batch size correct: 4
✓ Loaded all 3 config files
```

### Documentation

Created detailed documentation (`docs/dataset_factory.md`):
- Quick start guide
- Config file structure
- Registry API reference
- Extension examples
- Integration with training scripts
- CLI usage patterns
- Best practices

### Code Formatting

All new Python files reformatted with `black-with-tabs`:
- `trigor/data/registry.py`
- `test_dataset_factory.py`
- Tab indentation applied
- Consistent with project style

### Integration Points

**With existing framework:**
- Complements agent registry (`trigor/agents/registry.py`)
- Complements environment registry (`trigor/envs/registry.py`)
- Integrates with Hydra config system
- Works with existing TGNDataset and tokenizer

**For training:**
```python
# Full integration example
dataset = make_dataset(cfg.dataset.type, cfg.dataset)
env = make_env(cfg.env.type, cfg.env)
agent = make_agent(cfg.agent.type, env.observation_space, env.action_space, cfg.agent)
```

### Benefits

1. **Reproducibility** - Configs version-controlled, experiments reproducible
2. **Flexibility** - Easy to swap datasets via CLI without code changes
3. **Scalability** - Register new dataset types without modifying core code
4. **Discoverability** - `list_datasets()` shows available types
5. **Documentation** - Config files serve as usage documentation
6. **Type safety** - Factory validates before instantiation

### Next Steps

Potential enhancements:
- Add train/validation split configs
- Multi-dataset configs for combined loading
- Dataset preprocessing pipelines
- Custom filter functions in config
- Dataset caching strategies

</details>
