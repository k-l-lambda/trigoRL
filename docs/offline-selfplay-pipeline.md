# Offline Self-Play Training Pipeline

Complete end-to-end pipeline for training Trigo AI with C++ self-play data generation.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE TRAINING PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

    C++ Self-Play Generator              Python Training
    ───────────────────────              ───────────────

    ┌──────────────┐                    ┌──────────────┐
    │   Policy     │                    │   Dataset    │
    │  Interface   │                    │   Loader     │
    │              │                    │              │
    │  - Random    │                    │ TGNDataset   │
    │  - ONNX (*)  │                    │ + Tokenizer  │
    │  - MCTS (*)  │                    │              │
    └──────┬───────┘                    └──────▲───────┘
           │                                   │
           │ generates games                   │ reads TGN
           ▼                                   │
    ┌──────────────┐                    ┌──────┴───────┐
    │ TrigoGame    │─────────────────→  │  .tgn files  │
    │   Engine     │  exports TGN       │   (games)    │
    └──────────────┘                    └──────────────┘
           │                                   │
           │                                   │ loads
           ▼                                   ▼
    ┌──────────────┐                    ┌──────────────┐
    │   Game       │                    │  DataLoader  │
    │  Recorder    │                    │   (batches)  │
    └──────────────┘                    └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  Transformer │
                                        │     Model    │
                                        │   (GPT-2)    │
                                        └──────┬───────┘
                                               │
                                               │ trains
                                               ▼
                                        ┌──────────────┐
                                        │   Trained    │
                                        │   Weights    │
                                        └──────┬───────┘
                                               │
                                               │ exports
                                               ▼
                                        ┌──────────────┐
                                        │     ONNX     │─────┐
                                        │    Model     │     │
                                        └──────────────┘     │
                                                             │
                        ┌────────────────────────────────────┘
                        │ loop back for better data
                        ▼
                 (*) Future: Use trained ONNX model
                     for next generation of games
```

## Quick Start

### 1. Generate Self-Play Data

```bash
cd /home/camus/work/trigo.cpp/build

# Generate 1000 random games for training
./self_play_generator \
    --num-games 1000 \
    --board 5x5x5 \
    --black-policy random \
    --white-policy random \
    --output /path/to/training/data \
    --seed 42

# Output: 1000 .tgn files, ~3.33 games/sec
```

### 2. Train Model

```bash
cd /home/camus/work/trigoRL

# Train with default config
python train_lm.py configs/training/trigo-selfplay.yaml

# Or with custom overrides
python train_lm.py configs/training/trigo-selfplay.yaml \
    data.data_dir=/path/to/training/data \
    training.epochs=20 \
    training.learning_rate=1e-4
```

### 3. Export to ONNX (Future)

```bash
# Export trained model for C++ inference
python export_onnx.py outputs/trigor/20251205-trigo-selfplay/ \
    --output models/trigo_policy.onnx
```

### 4. Generate Better Data (Future)

```bash
# Use trained ONNX model for next generation
./self_play_generator \
    --num-games 1000 \
    --black-policy neural \
    --white-policy neural \
    --model models/trigo_policy.onnx \
    --output /path/to/better/data
```

## Architecture

### C++ Components

#### Policy Interface (`include/self_play_policy.hpp`)

```cpp
class IPolicy {
public:
    virtual PolicyAction select_action(const TrigoGame& game) = 0;
    virtual std::string name() const = 0;
};

// Current: Random baseline
class RandomPolicy : public IPolicy { ... };

// Future: ONNX model inference
class NeuralPolicy : public IPolicy {
    OrtSession* onnx_session;
    PolicyAction select_action(...);
};

// Future: MCTS tree search
class MCTSPolicy : public IPolicy { ... };
```

#### Game Recorder (`include/game_recorder.hpp`)

```cpp
struct SelfPlayRecord {
    BoardShape board_shape;
    std::string black_player, white_player;
    GameResult result;
    TerritoryResult final_territory;
    std::vector<Step> steps;  // Complete move history
};

class GameRecorder {
    static SelfPlayRecord record_game(const TrigoGame& game, ...);
    static std::string to_tgn(const SelfPlayRecord& record);
    static bool save_tgn(const SelfPlayRecord& record, const std::string& filename);
};
```

#### Self-Play Generator (`src/self_play_generator.cpp`)

Main executable with command-line interface for data generation.

### Python Components

#### Dataset Loader (`trigor/data/tgn_dataset.py`)

```python
dataset = TGNDataset(
    data_dir="/path/to/data",
    tokenizer=TGNByteTokenizer(),
    max_length=8192,
    split="*0..7/10",  # 80% train, deterministic hash-based split
)

# Outputs: {input_ids, labels, attention_mask}
```

#### Training Script (`train_lm.py`)

Hydra-based training with wandb integration:

```python
python train_lm.py configs/training/trigo-selfplay.yaml
```

## Configuration

### Training Config (`configs/training/trigo-selfplay.yaml`)

```yaml
data:
  type: TGNDataset
  data_dir: /tmp/selfplay_test
  max_length: 8192
  train_split: "*0..7/10"  # 80% train (shuffled)
  val_split: "8,9/10"      # 20% val (not shuffled)

  loader:
    batch_size: 2
    num_workers: 0

model:
  type: AttentionCausalLoss
  config:
    model_config:
      type: GPT2CausalLM
      config:
        vocab_size: 128      # TGN tokenizer
        hidden_size: 128
        num_layers: 4
        num_heads: 4
        max_seq_len: 8192

training:
  epochs: 5
  learning_rate: 3e-4
  weight_decay: 0.01
  dtype: bfloat16

  wandb:
    enabled: true
    tags:
      - cpp-selfplay
      - offline-training
```

## Testing

### Verify Dataset Loading

```bash
python tests/test_dataset_loading.py
```

**Expected output:**
- ✓ 20 TGN files loaded
- ✓ Tokenization working
- ✓ Train/val split correct
- ✓ No data leakage

### Verify Training Pipeline

```bash
python tests/test_training_pipeline.py
```

**Expected output:**
- ✓ Model created (544K params)
- ✓ Training loss: 4.80 → 4.10
- ✓ Validation loss: ~4.01
- ✓ No NaN or explosion

## Performance

### Current Performance

- **Generation:** 3.33 games/sec (random CPU policy)
- **Avg game length:** 200 moves
- **Output size:** ~1.4KB per game (TGN text format)

### Future Optimizations

| Policy Type | Speed (games/sec) | Description |
|-------------|-------------------|-------------|
| Random (current) | 3.33 | CPU baseline |
| ONNX Neural | 2-3 | NN inference overhead |
| CPU MCTS | 5-10 | Parallel tree search |
| CUDA MCTS | 50+ | GPU acceleration (20-50x) |

## TGN Format

### Example Output

```tgn
[Event "Self-Play Training"]
[Site "Trigo Self-Play"]
[Date "2025.12.05"]
[Black "Random"]
[White "Random"]
[Board 5x5x5]

1. ybz azz
2. zbb aaa
3. aba zzy
4. yzb bba
5. y0z Pass
...
13. Pass ; -2
```

### Format Specifications

- **Coordinates:** ab0yz encoding (center-symmetric)
  - `0` = center on each axis
  - `a,b,c,...` from one edge
  - `z,y,x,...` from opposite edge
- **Metadata:** Standard TGN tags
- **Score:** Final comment with score difference

## Tokenization

### TGN Byte Tokenizer

```python
# Vocabulary: 128 tokens
# 0-3: Special (PAD, START, END, VALUE)
# 10: Newline (LF)
# 32-127: ASCII printable (direct mapping)

tokenizer = TGNByteTokenizer()
tokens = tokenizer.encode(tgn_text, max_length=8192)
# Output: [START, ...tokens..., END, PAD, PAD, ...]
```

## Next Steps

### Immediate (Week 1-2)

1. Generate 10K random games for baseline training
2. Train small GPT-2 model (128 hidden, 4 layers)
3. Verify loss converges and model learns patterns
4. Export trained model to ONNX format

### Near-term (Week 3-4)

5. Implement `NeuralPolicy` with ONNX Runtime
6. Generate games with trained policy
7. Train improved model on better data
8. Iterate and compare Elo ratings

### Long-term (Month 2-3)

9. Implement CPU MCTS policy
10. Implement CUDA MCTS for 20-50x speedup
11. Add value head for position evaluation
12. Implement AlphaZero-style training

## Troubleshooting

### C++ Generation Issues

**Problem:** Compilation errors
```bash
cd /home/camus/work/trigo.cpp/build
cmake .. && make -j8
```

**Problem:** Slow generation
- Current: 3.33 games/sec is expected for random policy
- Future: Use CUDA MCTS for 50+ games/sec

### Python Training Issues

**Problem:** Dataset not found
```bash
# Check data directory exists
ls -lh /tmp/selfplay_test/*.tgn

# Generate data if missing
cd /home/camus/work/trigo.cpp/build
./self_play_generator --num-games 20 --output /tmp/selfplay_test
```

**Problem:** CUDA out of memory
```yaml
# Reduce batch size in config
data:
  loader:
    batch_size: 1  # Reduce from 2

model:
  config:
    model_config:
      config:
        hidden_size: 64  # Reduce from 128
```

**Problem:** Loss is NaN
- Check learning rate (try 1e-5 instead of 3e-4)
- Disable mixed precision: `training.dtype: float32`
- Check for data issues (corrupted TGN files)

## Files Overview

### C++ Files (trigo.cpp)

```
include/
├── self_play_policy.hpp     # Policy interface (230 lines)
├── game_recorder.hpp         # TGN export (276 lines)
└── trigo_game.hpp           # Game engine

src/
├── self_play_generator.cpp  # Main executable (254 lines)
└── ...

CMakeLists.txt               # Build config
```

### Python Files (trigoRL)

```
trigor/
├── data/
│   ├── tgn_dataset.py       # Dataset loader (322 lines)
│   ├── tokenizer.py         # TGN tokenizer (254 lines)
│   └── utils.py             # Split utils (139 lines)
├── models/
│   └── gpt2CausalLM.py     # GPT-2 wrapper
└── training/
    └── lm_trainer.py        # Training loop

configs/training/
└── trigo-selfplay.yaml      # Training config (110 lines)

tests/
├── test_dataset_loading.py       # Dataset tests (218 lines)
└── test_training_pipeline.py     # E2E tests (209 lines)

train_lm.py                  # Main entry point
```

## Resources

- **Project Root:** `/home/camus/work/trigoRL`
- **C++ Engine:** `/home/camus/work/trigo.cpp`
- **Test Data:** `/tmp/selfplay_test` (20 games)
- **Config Docs:** `configs/training/README.md` (if exists)
- **Development Log:** `agentlog.md`

## Support

For issues or questions:
1. Check test scripts: `tests/test_*.py`
2. Review agentlog.md for development history
3. Examine example configs in `configs/training/`
4. Run verification tests before full training
