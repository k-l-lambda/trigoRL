# Tinker API Analysis for TrigoRL

**Date**: 2025-12-21
**Purpose**: Evaluate whether Tinker API can improve TrigoRL's reinforcement learning workflow

---

## Executive Summary

**Recommendation**: **NOT RECOMMENDED** for TrigoRL at this stage.

**Key Findings**:
- ❌ **Game-specific RL mismatch**: Tinker is designed for LLM text generation tasks (math, reasoning, instruction-following), not board game environments
- ❌ **Custom environment complexity**: Trigo's 3D Go mechanics don't fit Tinker's token-based environment model
- ❌ **Cost concerns**: Hosted service pricing ($0.40/M tokens for 8B model) vs free local GPUs
- ❌ **Architecture limitations**: LoRA-only, no full fine-tuning, no custom model architectures
- ✅ **Good for**: LLM text tasks, but TrigoRL needs game-state→action RL

---

## What is Tinker?

### Overview
Tinker is a **hosted training API** by Thinking Machines Lab that simplifies LLM fine-tuning:
- **You write**: Python training loops on CPU with custom data/environments
- **Tinker handles**: Distributed GPU training, infrastructure, scaling

### Key Characteristics
| Aspect | Details |
|--------|---------|
| **Service Type** | Hosted cloud API (no self-hosting) |
| **Training Method** | LoRA fine-tuning only (no full fine-tuning) |
| **Pricing Model** | Pay per million tokens processed |
| **Target Use Case** | LLM post-training (instruction-following, reasoning, RLHF) |
| **Infrastructure** | Multi-tenant shared GPU pools |

---

## Tinker's RL Capabilities

### 1. Supported RL Methods

**RLVR (RL with Verifiable Rewards)**:
- Programmatic reward functions
- Checks model outputs against reference answers or unit tests
- Use cases: Math problems, code generation, reasoning

**RLHF (RL on Human Feedback)**:
- Train preference models via supervised learning
- Use preference model as reward function
- Optimize policy through self-play

### 2. Environment Model

**Token-Based Environments**:
```python
class Env:
    async def initial_observation() → (Observation, StopCondition)
    async def step(action: Action) → StepResult
```

- **Observations**: Token sequences (not game states)
- **Actions**: Generated text tokens
- **Rewards**: Computed from text outputs

**Example Use Cases**:
- GSM8K math problems: Model generates solution, reward = 1[correct answer]
- Twenty Questions: Model asks questions, reward based on success
- Tool use: Model generates API calls, reward from execution results

### 3. Loss Functions

- **Importance Sampling**: Basic policy gradient with importance weights
- **PPO**: Proximal Policy Optimization with clipped ratio
- **CISPO**: Clipped IS policy optimization
- **DRO**: Direct Reward Optimization with quadratic penalty
- **Custom losses**: Via `forward_backward_custom`

### 4. Technical Architecture

**Training Flow**:
```
1. Your CPU script samples: model → generate tokens
2. Your code computes: rewards from outputs
3. Tinker trains: forward_backward() → optim_step()
4. Repeat
```

**Clock Cycle System**:
- Multi-tenant shared infrastructure
- Jobs time-share GPU pools
- Async batching critical for efficiency

---

## TrigoRL's Current Workflow

### Architecture Overview

**Tech Stack**:
- **Models**: Custom PyTorch implementations (GPT-2, LLaMA, RWKV, xLSTM)
- **Training**: Local GPUs with PyTorch + Transformers
- **Data**: TGN (Trigo Game Notation) format
- **Environment**: C++ game engine with Python bindings (planned)
- **Codebase**: ~6,000 lines of custom Python

### Training Pipeline (Current/Planned)

```
1. Self-Play Generation:
   - C++ MCTS with neural network guidance (trigo.cpp)
   - Generates TGN game files
   - Stores position-value pairs

2. Supervised Learning:
   - Train value networks on self-play data
   - Custom TGNDataset (tokenizer for board states)
   - Standard PyTorch training loop

3. RL Training (Planned):
   - Episode-based RL with custom TrigoEnv
   - Policy optimization via game outcomes
   - Full model fine-tuning
```

### Custom Components

**Data Layer**:
- `TGNDataset`: Tokenizes 3D board states as sequences
- `TGNValueDataset`: Position-value pairs for value network
- Custom tokenizer for board representation

**Model Layer**:
- 4 custom model architectures (GPT-2, LLaMA, RWKV, xLSTM)
- Value heads for position evaluation
- Policy heads for move selection (planned)

**Training Layer**:
- `LMTrainer`: Language model training
- `RLTrainer`: RL episode-based training (skeleton exists)
- Checkpoint management, wandb logging

**Game Integration**:
- `trigo.cpp`: C++ MCTS implementation
- ONNX export for inference
- TGN format for game notation

---

## Fit Analysis: Tinker vs TrigoRL

### ❌ Major Mismatches

#### 1. **Environment Model Incompatibility**

**Tinker's Model**:
- Token-based text generation
- Environment gives text prompts
- Model generates text responses
- Rewards computed from text outputs

**TrigoRL's Needs**:
- 3D board state representation
- Valid move selection from legal moves
- Rewards from game outcomes (win/loss/draw)
- Multi-step episodic gameplay

**Why It Doesn't Fit**:
```python
# Tinker expects:
observation = "Solve: 2 + 3 = ?"
action = model.generate("5")
reward = 1 if action == "5" else 0

# TrigoRL needs:
observation = Board3D(5x5x5)  # 125 positions
action = Position(x=2, y=3, z=1)  # One of ~125 legal moves
reward = +1 if game.winner == Black else -1  # Only at end of game
```

**Workarounds Are Painful**:
- Encode board as text (loses spatial structure)
- Map text output to moves (error-prone parsing)
- Reward shaping becomes complex
- No natural fit for MCTS integration

#### 2. **Cost Structure Mismatch**

**Tinker Pricing (USD per million tokens)**:
| Model | Training Cost |
|-------|---------------|
| Llama-3.2-1B | $0.09 |
| Qwen3-4B-Instruct | $0.22 |
| Llama-3.1-8B | $0.40 |
| Llama-3.1-70B | $3.16 |

**Cost Estimate for TrigoRL**:
```
Assumptions:
- 1 game = 50 moves average
- 1 move = 200 tokens (board state + history)
- 1 self-play iteration = 1000 games
- Total = 50 × 200 × 1000 = 10M tokens

Cost per iteration (8B model):
10M tokens × $0.40/M = $4.00

For 100 iterations = $400
For 1000 iterations = $4000
```

**TrigoRL Current Cost**:
- **Local GPU**: Free (assuming existing hardware)
- **Or cloud GPU**: ~$0.50/hour (A100), ~$50 for 100 hours
- **Advantage**: Can train indefinitely without per-token charges

#### 3. **Architecture Limitations**

**Tinker Constraints**:
| Feature | Tinker | TrigoRL Needs |
|---------|--------|---------------|
| Training method | LoRA only | Full fine-tuning |
| Model architectures | Fixed (Qwen, LLaMA, etc.) | Custom (GPT-2, RWKV, xLSTM) |
| Model size | Minimum 1B parameters | Want to test smaller models |
| Infrastructure | Hosted cloud | Local control |

**Why This Matters**:
- TrigoRL has 4 custom model architectures
- Need to experiment with small models (<1B)
- Want full parameter fine-tuning, not adapter layers
- Existing codebase is PyTorch-native

#### 4. **Integration Complexity**

**What Needs to Change**:
1. **Rewrite environment**: Convert board states to token sequences
2. **Rewrite actions**: Parse text outputs as moves
3. **Abandon C++ MCTS**: Can't integrate with token-based API
4. **Rewrite reward logic**: Map game outcomes to text generation rewards
5. **Abandon custom models**: Use only Tinker's supported models
6. **Rewrite data pipeline**: TGN → text format conversion

**Estimated Effort**: 2-4 weeks of full-time work

**Risk**: High chance of ending up with a worse system due to impedance mismatches

### ✅ Minor Advantages

#### 1. **Infrastructure Abstraction**
- Don't need to manage distributed training
- Handles GPU allocation, failure recovery
- Scales to large models (70B+) without local hardware

**But**: TrigoRL doesn't need 70B models; 1-8B is sufficient for board games

#### 2. **Standard RL Algorithms**
- PPO, CISPO, DRO implementations provided
- Tested on benchmark tasks

**But**: TrigoRL already plans standard algorithms; implementation isn't hard

#### 3. **Async Training**
- Efficient pipelining with async requests
- Clock cycle optimization

**But**: Local training with PyTorch is already efficient

---

## Alternative: Keep Current Workflow

### Advantages of Local Training

**1. Perfect Fit for Game RL**:
```python
# Natural game environment
class TrigoEnv:
    def reset() → Board3D
    def step(action: Position) → (Board3D, reward, done, info)
    def valid_moves() → List[Position]
```

**2. Full Control**:
- Custom model architectures
- Any training algorithm (PPO, A2C, AlphaZero)
- Flexible reward shaping
- Integration with C++ MCTS

**3. Cost Effective**:
- Free if using existing GPUs
- Pay once for cloud GPUs, train unlimited iterations
- No per-token pricing

**4. Leverages Existing Code**:
- 6,000 lines of working code
- TGN format and tokenizer
- Custom models already implemented
- C++ MCTS integration

### Recommended Improvements (Without Tinker)

**1. Complete RL Trainer**:
```python
# Finish trigor/training/rl_trainer.py
class RLTrainer:
    def train_episode(self):
        # Self-play with MCTS
        # Collect trajectories
        # Compute advantages
        # Update policy and value networks
```

**2. Integrate trigo.cpp**:
- Python bindings for C++ MCTS
- Use ONNX models for inference
- Generate high-quality self-play data

**3. Implement Standard RL**:
- PPO or A2C for policy optimization
- Value network for position evaluation
- Advantage estimation (GAE)

**4. Optional: Cloud GPUs if Needed**:
- Use AWS/GCP spot instances
- Still cheaper than Tinker for your use case
- Full control over training

---

## When Tinker WOULD Make Sense

### Good Use Cases (Not TrigoRL)

**1. LLM Reasoning Tasks**:
```python
# Teaching models to solve math problems
env = "Solve: What is 15% of 80?"
model_output = "15% of 80 is 12"
reward = 1 if correct else 0
```

**2. Instruction Following**:
```python
# RLHF for chat models
preferred = "Here's a helpful answer..."
rejected = "I don't know"
# DPO or RLHF training
```

**3. Tool Use**:
```python
# Teaching models to use APIs
observation = "User wants weather in SF"
action = "get_weather(city='San Francisco')"
reward = 1 if tool_call_succeeds else 0
```

**4. Large Model Fine-Tuning**:
- Need 70B+ models for complex reasoning
- Don't have local GPU infrastructure
- Per-token cost justified by task value

### Why TrigoRL Doesn't Fit

| Tinker Sweet Spot | TrigoRL Reality |
|-------------------|-----------------|
| Text generation | Board state representation |
| Single-turn outputs | Multi-step episodes |
| Large models (70B+) | Small models (1-8B) |
| Standard architectures | Custom architectures |
| Pay per use | Need unlimited training |

---

## Cost-Benefit Analysis

### Tinker Option

**Costs**:
- **Money**: $4/iteration × 1000 iterations = $4,000
- **Time**: 2-4 weeks rewriting everything
- **Risk**: High chance of worse performance
- **Flexibility**: Locked into Tinker's models and APIs

**Benefits**:
- **Infrastructure**: Don't manage GPUs
- **Scaling**: Can use 70B models (but don't need them)

### Local Training Option

**Costs**:
- **GPU**: Free (existing) or $50-500 (cloud spot instances)
- **Time**: 1-2 weeks finishing RL trainer
- **Maintenance**: Manage training scripts

**Benefits**:
- **Perfect fit**: Native game environment support
- **Full control**: Any model, any algorithm
- **Cost-effective**: Unlimited training iterations
- **Leverage existing work**: 6,000 lines of code

**Winner**: **Local training** - better fit, lower cost, faster to market

---

## Technical Deep Dive: Why the Mismatch?

### Fundamental Difference: Text vs Structured Actions

**LLM Tasks (Tinker's Domain)**:
```
State space: Arbitrary text strings
Action space: Next token prediction (vocabulary size ~50k)
Trajectory: Sequence of generated tokens
Reward: Evaluated from complete text output

Example:
Observation: "What is 2+2?"
Action: Generate tokens ["4"]
Reward: 1.0 (correct answer)
```

**Board Game Tasks (TrigoRL's Domain)**:
```
State space: Discrete board configurations (3^125 for 5×5×5)
Action space: Valid moves only (~10-125 per position)
Trajectory: Sequence of game states and actions
Reward: Win/loss/draw at end of game

Example:
Observation: Board3D with stones at positions (...)
Action: place_stone(x=2, y=3, z=1)
Reward: +1.0 (only if game ends in win)
```

### Why Token-Based RL Doesn't Work for Games

**1. Representation Loss**:
```python
# Option A: Encode board as text
board_text = "Board: (0,0,0)=Black, (1,1,1)=White, ..."
# Loses spatial structure, attention patterns wrong
```

**2. Action Space Mismatch**:
```python
# Model generates text: "Play at position 2 3 1"
# Need to parse, validate, map to internal representation
# Error-prone, slow, unnatural
```

**3. Reward Timing**:
```python
# LLM tasks: Immediate reward per generated text
reward = check_answer(generated_text)

# Game tasks: Delayed reward at game end
reward = 0 for all moves except last
# Requires credit assignment across 50+ moves
```

**4. MCTS Integration Impossible**:
```python
# TrigoRL wants:
mcts_value = neural_network(board_state)
best_move = mcts.search(board, value_fn=neural_network)

# Tinker gives:
text_output = model.generate(text_prompt)
# Can't plug into MCTS tree search
```

---

## Conclusion

### Summary of Findings

**Tinker is a powerful tool for LLM post-training**, but **completely mismatched for TrigoRL**:

| Factor | Assessment | Impact |
|--------|-----------|--------|
| Environment fit | ❌ Poor | High |
| Cost structure | ❌ Expensive | High |
| Architecture | ❌ Limiting | High |
| Integration | ❌ Complex | High |
| Infrastructure | ✅ Good | Low (don't need it) |

### Final Recommendation

**DO NOT use Tinker for TrigoRL**

**Instead**:
1. ✅ Complete the existing `RLTrainer` implementation
2. ✅ Integrate `trigo.cpp` C++ MCTS via Python bindings
3. ✅ Implement PPO/A2C with custom game environment
4. ✅ Use local GPUs or cheap cloud spot instances
5. ✅ Leverage existing 6,000 lines of working code

**Estimated time to working RL pipeline**:
- **With Tinker**: 2-4 weeks + ongoing costs + worse performance
- **Without Tinker**: 1-2 weeks + one-time GPU cost + better performance

### When to Revisit Tinker

Consider Tinker in the future if:
- ❓ You want to train a **language model to describe Trigo strategies** (text generation task)
- ❓ You need to scale to **70B+ models** for complex reasoning about game positions
- ❓ You decide to pivot to **LLM-based game commentary or analysis**

But for **actual game-playing RL**, stick with traditional game RL infrastructure.

---

## References

### Tinker Documentation
- [Main Site](https://thinkingmachines.ai/tinker/)
- [Documentation](https://tinker-docs.thinkingmachines.ai/)
- [GitHub - tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [RL Overview](https://tinker-docs.thinkingmachines.ai/rl)
- [Model Lineup](https://tinker-docs.thinkingmachines.ai/model-lineup)
- [Under the Hood](https://tinker-docs.thinkingmachines.ai/under-the-hood)

### TrigoRL Resources
- [Project README](/home/camus/work/trigoRL/README.md)
- [CLAUDE.md](/home/camus/work/trigoRL/CLAUDE.md)
- [RL Trainer](/home/camus/work/trigoRL/trigor/training/rl_trainer.py)
- [trigo.cpp MCTS](/home/camus/work/trigo.cpp/include/mcts.hpp)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-21
**Reviewer**: Claude (Sonnet 4.5)
