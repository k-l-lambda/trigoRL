# TrigoRL

A reinforcement learning laboratory project for training AI agents to play Trigo, a 3D variant of the board game Go.

## Overview

TrigoRL is an experimental platform for exploring reinforcement learning techniques in the context of **Trigo**
- a strategic board game that extends the rules of Go into three-dimensional space.
While traditional Go is played on a 2D 19×19 board, Trigo is played on a cubic grid,
introducing new strategic dimensions and complexity.

## About Trigo

Trigo is a modern reimplementation of a 3D Go variant with the following characteristics:

- **Board**: 3D cubic grid (default: 5×5×5, configurable to other dimensions including 2D boards)
- **Rules**: Based on Go mechanics adapted for 3D space
  - Stone placement with capture detection
  - Ko rule enforcement
  - Territory calculation in 3D
  - Pass, undo/redo, and resignation support
- **Notation**: TGN (Trigo Game Notation) - a PGN-inspired text format for recording games
- **Coordinate System**: Center-symmetric notation (e.g., `000` = center, `aaa` = corner)

**TRY IT YOURSELF ONLINE**: here is a [Trigo demo page](https://huggingface.co/spaces/k-l-lambda/trigo).

## Quick Start

### Inspect Dataset

View and validate the TGNDataset:

```bash
# View dataset statistics
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --stats

# Validate dataset implementation
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --validate

# View a specific sample
python tools/view_dataset.py configs/training/trigo-gpt2.yaml --sample 0 --tokens
```

See [tools/README.md](tools/README.md) for comprehensive CLI documentation.

### Test Models

Run the model test suite:

```bash
python tests/test_models.py
```

This validates:
- Model registry with 4 CausalLM models
- Configuration loading (dict and OmegaConf)
- Forward passes for GPT-2, LLaMA, and RWKV
- Parameter counting and memory estimation

### Verify Configurations

Test all training configs:

```bash
python examples/verify_training_configs.py
```

### Export Models to ONNX

Export trained models for cross-platform deployment:

```bash
# Export best checkpoint (default - standard inference mode)
python exportOnnx.py outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt

# Export in evaluation mode with fixed dimensions
python exportOnnx.py outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt \
    --evaluation-mode --prefix-len 10 --seq-len 15

# Export evaluation mode with dynamic dimensions (prefix-len/seq-len only for dummy input)
python exportOnnx.py outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt \
    --evaluation-mode --dynamic-seq

# Export with INT8 quantization (recommended for deployment)
python exportOnnx.py outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt \
    --quantize --quant-type int8

# Export with dynamic batch/sequence sizes
python exportOnnx.py outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt \
    --dynamic-batch --dynamic-seq

# Export with static quantization (best accuracy)
python exportOnnx.py outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt \
    --quantize --quant-method static --calibration-samples 200
```

**Export Modes**:
- **Standard mode**: Single input `input_ids`, returns `logits` for all positions
- **Evaluation mode**: Three inputs (`prefix_ids`, `evaluated_ids`, `evaluated_mask`), returns logits for last prefix + evaluated positions. Supports custom attention patterns like tree attention for computing sequence probabilities.

**Quantization benefits**:
- **INT8 dynamic**: ~3-4x smaller model, minimal accuracy loss
- **INT4**: ~8x smaller, more aggressive compression
- **Static quantization**: Better accuracy than dynamic, requires calibration

See `docs/onnx_quantization_guide.md` for comprehensive quantization documentation.

## Technical Stack

### Reinforcement Learning Framework

- **PyTorch**: Deep learning framework for model implementation
- **Transformers**: Architecture foundation for the RL agent (GPT-2, LLaMA, RWKV, xLSTM)
- **Weights & Biases (wandb)**: Training metrics and experiment tracking
- **ONNX**: Model weight export format for cross-platform deployment
- **OmegaConf/Hydra**: Hierarchical configuration management

### Current Implementation Status

✅ **Data Pipeline**
- TGNDataset: PyTorch dataset for TGN files with byte-level tokenization
- TGNByteTokenizer: 259-token vocab (256 bytes + PAD/START/END)
- Configuration-driven dataset loading

✅ **Model Architecture**
- 4 CausalLM models: GPT2, LLaMA (with GQA), RWKV (linear attention), xLSTM
- Model registry with factory pattern
- OmegaConf integration for flexible configuration
- Parameter counting and memory footprint estimation

✅ **Training Configuration**
- Complete YAML configs for all 4 models
- Hyperparameters tuned for each architecture
- WandB integration (optional)
- Checkpointing and learning rate scheduling

✅ **Development Tools**
- CLI tool for dataset inspection and validation
- Model testing suite (109 tests passing)
- Configuration verification scripts

✅ **Model Export**
- ONNX export script with checkpoint loading
- INT8/INT4 quantization (dynamic and static)
- 3-4x model compression with minimal accuracy loss
- Node.js inference validation and testing

## Development Roadmap

The following components need to be implemented for the RL framework:

1. ~~**Data Pipeline**~~ ✅ COMPLETE
   - ~~TGNDataset implementation with byte tokenization~~
   - ~~Dataset configuration and loading~~
   - ~~Validation and inspection tools~~

2. ~~**Model Architecture**~~ ✅ COMPLETE
   - ~~Transformer-based CausalLM implementations~~
   - ~~Model registry and factory pattern~~
   - ~~Configuration management~~

3. **Training Pipeline** 🚧 IN PROGRESS
   - Training loop implementation
   - Self-play game generation
   - Experience replay buffer
   - Policy gradient or actor-critic implementation
   - Integration with Weights & Biases for experiment tracking

4. **Environment Wrapper** 📋 PLANNED
   - Python interface to the Trigo game engine
   - OpenAI Gym-compatible environment
   - State representation for 3D board positions
   - Action space definition

5. ~~**Model Export**~~ ✅ COMPLETE
   - ~~ONNX conversion utilities~~
   - ~~INT8/INT4 quantization support~~
   - ~~Static and dynamic quantization~~
   - ~~Node.js inference validation~~

6. **Evaluation & Analysis** 📋 PLANNED
   - Agent performance metrics
   - Game quality assessment
   - Visualization tools

## Game Engine Features

The Trigo game engine provides:

- **3D Visualization**: Interactive Three.js-based board rendering
- **Multiplayer Support**: Real-time gameplay via WebSocket
- **Game Notation**: TGN format for saving and loading games
- **REST API**: Programmatic game control
- **Comprehensive Testing**: 10 test suites covering core functionality

For detailed API documentation, see:
- [Game Engine README](third_party/trigo/README.md)
- [TGN Format Specification](third_party/trigo/docs/tgn-format-spec.md)
- [Development Guidelines](third_party/trigo/CLAUDE.md)

## Acknowledgments

- Based on the Trigo game engine by k-l-lambda
- Inspired by AlphaGo and other game-playing RL systems
