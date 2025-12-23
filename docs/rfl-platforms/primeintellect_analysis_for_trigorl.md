# PrimeIntellect Analysis for TrigoRL

**Date**: 2025-12-21
**Purpose**: Evaluate whether PrimeIntellect can improve TrigoRL's reinforcement learning workflow

---

## Executive Summary

**Recommendation**: **PARTIALLY SUITABLE** - PrimeIntellect offers valuable infrastructure but requires custom environment development.

**Key Findings**:
- ✅ **Flexible compute platform**: Pay-as-you-go GPU access, no vendor lock-in
- ✅ **Custom RL environments**: Can create game-specific environments with `verifiers`
- ✅ **Cost competitive**: $0.79-1.49/hr for A100/H100 vs $2-8/hr elsewhere
- ⚠️ **Setup investment**: Need to build custom Trigo environment from scratch
- ⚠️ **LLM-focused**: Framework designed for language models, needs adaptation for games
- ✅ **Better than Tinker**: More control, better fit for custom tasks

**Best Use Case**: If you need scalable GPU compute for distributed training

---

## What is PrimeIntellect?

### Platform Type
**Multi-Cloud GPU Aggregator + RL Training Platform**

PrimeIntellect is fundamentally different from Tinker:
- **Infrastructure Layer**: Aggregates GPUs across multiple cloud providers
- **Training Framework**: Provides `prime-rl` for large-scale async RL
- **Environment Hub**: Community platform for sharing RL environments
- **Self-Hosted**: You get full SSH access and run your own code

### Core Services

| Service | Description |
|---------|-------------|
| **On-Demand GPU Cloud** | Access GPUs across providers (AWS, GCP, Lambda, community) |
| **Multi-Node Clusters** | Scale up to 256 H100s with FSDP training |
| **Environments Hub** | Share and discover RL environments |
| **prime-rl** | Async RL training framework (FSDP2 + vLLM) |
| **verifiers** | Library for building RL environments |
| **Inference API** | Optional hosted inference service |
| **Storage** | Persistent and cluster storage options |

---

## Detailed Analysis: prime-rl Framework

### Architecture

**Three-Component System**:
```
┌─────────────┐      ┌──────────────┐      ┌─────────┐
│  Inference  │◄────►│ Orchestrator │◄────►│ Trainer │
│   (vLLM)    │      │              │      │ (FSDP2) │
└─────────────┘      └──────────────┘      └─────────┘
     │                       │                    │
     └───────────────────────┴────────────────────┘
              Async RL Training Loop
```

**Components**:
1. **Trainer**: FSDP2-based distributed training
   - Supports SFT (supervised fine-tuning) and RL
   - Full fine-tuning or LoRA
   - Multi-node scaling

2. **Inference Server**: vLLM-powered serving
   - Generates model responses during rollouts
   - Batched inference for efficiency
   - Async operation

3. **Orchestrator**: Coordination layer
   - Manages data flow between components
   - Handles environment interactions
   - Collects trajectories and computes rewards

### Training Pipeline

**Phase 1: SFT (Optional)**
```bash
uv run sft @ configs/<env>/sft.toml
```
- Supervised fine-tuning on seed data
- Warm-start for RL training
- Standard next-token prediction

**Phase 2: RL Training**
```bash
uv run rl \
  --trainer @ configs/<env>/train.toml \
  --orchestrator @ configs/<env>/orch.toml \
  --inference @ configs/<env>/infer.toml
```

**Workflow**:
```
1. Inference server: Generate completions from prompts
2. Orchestrator: Run environment, compute rewards
3. Trainer: Collect trajectories, update policy
4. Repeat (async, decentralized)
```

---

## Verifiers: Environment Library

### What is Verifiers?

**Modular RL environment builder** for LLM agents:
- Python library for creating custom environments
- Multiple environment types for different tasks
- Integration with `prime-rl` and OpenAI-compatible APIs

### Environment Types

| Type | Description | Use Case |
|------|-------------|----------|
| `SingleTurnEnv` | One prompt → one response | Classification, QA |
| `MultiTurnEnv` | Interactive conversation | Dialogue, games |
| `ToolEnv` | Environment provides tools | API usage, code exec |
| `StatefulToolEnv` | Tools with persistent state | Complex workflows |
| `SandboxEnv` | Isolated execution | Code generation |
| `PythonEnv` | Python code execution | Programming tasks |

### Environment Structure

**Core Components**:
```python
import verifiers as vf
from datasets import Dataset

# 1. Define reward function
def reward_fn(prompt, completion, info) -> float:
    # Compute reward from completion
    return score

rubric = vf.Rubric(funcs=[reward_fn], weights=[1.0])

# 2. Create environment
env = vf.SingleTurnEnv(
    dataset=Dataset.from_dict({
        'prompt': [...],  # List of prompts
        'answer': [...],  # Optional ground truth
    }),
    rubric=rubric
)

# 3. Evaluate
results = await env.evaluate(
    client=AsyncOpenAI(),
    model="gpt-4",
    num_examples=100
)
```

### Example Environments

**Included in Repository**:
- **Wordle**: Word-guessing game with feedback
- **Wiki Search**: Wikipedia navigation with tools
- **Math Python**: Math problems with code execution
- **Reverse Text**: Simple text manipulation

**Community Hub**:
- Hundreds of pre-built environments
- Browse: https://app.primeintellect.ai/dashboard/environments

---

## GPU Pricing Comparison

### PrimeIntellect Pricing

| GPU Type | PrimeIntellect | Typical Market | Savings |
|----------|----------------|----------------|---------|
| **H100 SXM5** | $1.49/hr | $2.25-8.00/hr | 34-81% |
| **A100 80GB** | $0.79/hr | $1.29-4.00/hr | 39-80% |
| **A6000 48GB** | $0.41/hr | $0.49/hr | 16% |
| **RTX 4090** | $0.32/hr | $0.44/hr | 27% |

**Multi-Node Clusters**:
- **Premium H100 (InfiniBand)**: ~$52.80/hr (8x H100 + networking)
- **Value H100 (Ethernet)**: ~$40.80/hr

### Cost Estimation for TrigoRL

**Scenario 1: Local-Scale Training**
```
GPU: 1x A100 80GB
Duration: 100 hours of training
Cost: 100 × $0.79 = $79
```

**Scenario 2: Distributed Training**
```
GPU: 4x A100 80GB
Duration: 25 hours (4x parallelism)
Cost: 25 × 4 × $0.79 = $79
(Same total cost, 4x faster)
```

**Scenario 3: Large-Scale Experiment**
```
GPU: 8x H100 cluster
Duration: 10 hours
Cost: 10 × 8 × $1.49 = $119.20
(For heavy multi-node experiments)
```

### Comparison with Alternatives

| Option | Infrastructure | Cost Model | TrigoRL Estimate |
|--------|---------------|------------|------------------|
| **Tinker** | Hosted API | $0.40/M tokens | $400-4000 (1000 iterations) |
| **PrimeIntellect** | Self-managed VMs | $0.79/hr | $79 (100 hrs) |
| **Local GPU** | Own hardware | Free | $0 (if already owned) |
| **AWS/GCP On-Demand** | Major cloud | $2-4/hr | $200-400 (100 hrs) |
| **AWS/GCP Spot** | Spot instances | $0.80-1.50/hr | $80-150 (100 hrs) |

**Key Insight**: PrimeIntellect is **~50% cheaper** than major cloud on-demand, **similar** to spot pricing but more reliable.

---

## Fit Analysis for TrigoRL

### ✅ Advantages Over Tinker

| Factor | Tinker | PrimeIntellect | Winner |
|--------|--------|----------------|--------|
| **Environment Control** | Token-based only | Full SSH + custom code | ✅ PI |
| **Custom Models** | Tinker's models only | Any PyTorch model | ✅ PI |
| **Training Method** | LoRA only | Full fine-tuning + LoRA | ✅ PI |
| **Cost Structure** | Per-token pricing | Per-hour GPU | ✅ PI (for RL) |
| **Infrastructure Access** | No SSH, API only | Full root access | ✅ PI |
| **Game RL Support** | Poor fit | Possible with custom env | ✅ PI |

### ✅ Strengths for TrigoRL

#### 1. **Flexible Infrastructure**
- Get SSH access to VMs
- Install any dependencies
- Run custom training scripts
- No vendor lock-in

**Example**:
```bash
# SSH into PrimeIntellect GPU
ssh -i prime.pem ubuntu@<gpu-ip>

# Install TrigoRL
git clone https://github.com/you/trigoRL
cd trigoRL
pip install -e .

# Run training with your existing code
python train_lm.py configs/training/trigo-gpt2.yaml
```

#### 2. **Custom Environment Support**

**Can Build Trigo Environment**:
```python
# trigor_env.py
import verifiers as vf
from datasets import Dataset

class TrigoEnv(vf.MultiTurnEnv):
    """Trigo game environment for RL training."""

    def __init__(self, dataset, rubric, max_turns=50):
        super().__init__(dataset, rubric, max_turns)
        self.game_engine = TrigoGame()  # Your C++ bindings

    async def env_response(self, messages, state):
        # Parse model's move from last message
        move = self.parse_move(messages[-1])

        # Apply move to game
        self.game_engine.apply_move(move)

        # Get new board state
        board_state = self.game_engine.to_tgn()

        # Check if game ended
        if self.game_engine.is_finished():
            reward = self.compute_final_reward()
            return [], {"game_over": True, "reward": reward}

        # Return next prompt
        prompt = f"Current board:\n{board_state}\nYour move:"
        return [{"role": "system", "content": prompt}], state

    def parse_move(self, message):
        # Parse TGN notation from text
        ...

    def compute_final_reward(self):
        # +1 for win, -1 for loss
        ...
```

#### 3. **Distributed Training Capability**

**Multi-Node FSDP**:
```bash
# Deploy 4-node cluster on PrimeIntellect
# Each node: 8x A100

# Train large model with FSDP
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node-rank=$RANK \
    train_distributed.py
```

**Benefits**:
- Scale to 70B+ models if needed
- Faster training with parallelism
- Automatic sharding with FSDP

#### 4. **Cost Competitive**

**vs Tinker**:
```
Tinker: $0.40/M tokens × 10M tokens/iter × 1000 iter = $4,000
PrimeIntellect: $0.79/hr × 100 hrs = $79
Savings: 98%
```

**vs AWS On-Demand**:
```
AWS: $3.00/hr × 100 hrs = $300
PrimeIntellect: $0.79/hr × 100 hrs = $79
Savings: 74%
```

### ⚠️ Challenges and Limitations

#### 1. **Custom Environment Development Required**

**What Needs Building**:
```
┌─────────────────────────────────────────┐
│ Trigo Environment (New Development)     │
├─────────────────────────────────────────┤
│ 1. Text ←→ Board State Conversion       │
│    - Parse TGN from model output        │
│    - Encode board state as text         │
│                                         │
│ 2. Environment Logic                    │
│    - Game state management              │
│    - Legal move validation              │
│    - Reward computation                 │
│                                         │
│ 3. Integration with verifiers           │
│    - Implement MultiTurnEnv interface   │
│    - Handle async evaluation            │
│                                         │
│ 4. Testing and Debugging                │
│    - Validate environment correctness   │
│    - Test with dummy models             │
└─────────────────────────────────────────┘
```

**Estimated Effort**: 1-2 weeks

**Risk**: Still need text-based interface (model generates text → parse as moves)

#### 2. **LLM-Focused Architecture**

**Mismatch Areas**:
- `prime-rl` expects language models generating text
- Orchestrator assumes text completions
- Reward functions work on text outputs

**Adaptation Required**:
```python
# Current LLM pattern:
prompt = "Solve: 2+2=?"
completion = "4"  # Model generates text
reward = 1 if completion == "4" else 0

# Needed for Trigo:
prompt = "Board: [5x5x5 state]\nMove:"
completion = "aa0"  # Model generates TGN
reward = win_loss_at_game_end  # Only available after 50 moves
```

**Workaround**: Use text-based move notation (TGN) as interface

#### 3. **Async RL Design**

**prime-rl uses off-policy async training**:
- Inference server may lag behind trainer
- Trajectories collected from slightly old policy
- Works for LLM tasks, may need tuning for games

**For TrigoRL**:
- Games have long episodes (50+ moves)
- Off-policy may be acceptable
- Need to test stability

#### 4. **No MCTS Integration Out-of-the-Box**

**Your Current Workflow**:
```
C++ MCTS → Neural network value/policy → ONNX export
```

**PrimeIntellect Workflow**:
```
Python RL trainer → Model → Text generation
```

**Integration Challenge**:
- `prime-rl` doesn't have built-in MCTS
- Would need to add MCTS as custom component
- Or abandon MCTS, use pure policy gradient

---

## Use Case Scenarios

### ✅ Scenario 1: Scale Up Existing Training

**Situation**: Your local GPU (RTX 4090) is too slow

**Solution**:
```bash
# 1. Deploy A100 on PrimeIntellect
prime deploy --gpu a100 --count 1

# 2. SSH and run existing code
ssh -i prime.pem ubuntu@<gpu-ip>
git clone <your-trigoRL>
python train_lm.py configs/training/trigo-gpt2.yaml

# Cost: $0.79/hr
# Speed: ~3x faster than RTX 4090
```

**Verdict**: ✅ **Perfect fit** - No code changes needed

### ✅ Scenario 2: Distributed Training

**Situation**: Want to train 70B model with FSDP

**Solution**:
```bash
# Deploy 4x A100 cluster
prime deploy --gpu a100 --count 4 --cluster

# Run FSDP training
torchrun --nnodes=4 train_distributed.py

# Cost: 4 × $0.79 = $3.16/hr
```

**Verdict**: ✅ **Good fit** - Standard distributed training

### ⚠️ Scenario 3: RL Training with Custom Environment

**Situation**: Want to use `prime-rl` for RL training

**Solution**:
```bash
# 1. Build Trigo environment
prime env init trigor-trigo
# ... implement MultiTurnEnv ...
prime env push

# 2. Configure prime-rl
uv run rl @ configs/trigo/rl.toml

# Cost: Same as GPU cost
```

**Verdict**: ⚠️ **Possible but requires work** - 1-2 weeks to build environment

**Pros**:
- Leverage async RL infrastructure
- Scale to multi-GPU easily
- Community sharing of environment

**Cons**:
- Text-based interface (parse TGN from model output)
- Need to adapt reward computation
- Learning curve for `prime-rl` + `verifiers`

### ❌ Scenario 4: AlphaZero-Style Training

**Situation**: Want MCTS + policy + value network (AlphaZero)

**Solution**: Not directly supported

**Verdict**: ❌ **Poor fit** - `prime-rl` is designed for policy-only RL, not AlphaZero

**Alternative**: Use PrimeIntellect just for compute, run your own AlphaZero code

---

## Comparison Matrix

### PrimeIntellect vs Tinker vs Local

| Factor | PrimeIntellect | Tinker | Local GPU |
|--------|----------------|--------|-----------|
| **Cost (100 hrs)** | $79 (A100) | $4000 | Free |
| **Setup Time** | 5 minutes | 2-4 weeks | Already have |
| **Environment Control** | Full SSH access | API only | Full control |
| **Custom Models** | Any PyTorch | Tinker's only | Any PyTorch |
| **Training Method** | Full fine-tune + LoRA | LoRA only | Any method |
| **Scaling** | Multi-node FSDP | Handled by Tinker | Single GPU |
| **Game RL Support** | Possible (custom env) | Poor | Native |
| **MCTS Integration** | Manual | Impossible | Native |
| **Pay Model** | Per hour | Per token | One-time |
| **Best For** | Scale-up, multi-GPU | LLM text tasks | Development |

### Recommendation by Use Case

| Goal | Best Choice | Runner-Up |
|------|-------------|-----------|
| **Local development** | Local GPU | PrimeIntellect |
| **Scale to large models** | PrimeIntellect | - |
| **Distributed training** | PrimeIntellect | AWS/GCP |
| **Pure policy RL** | PrimeIntellect | Local |
| **AlphaZero/MCTS** | Local | PrimeIntellect (for compute) |
| **Budget < $100** | PrimeIntellect | Tinker ❌ |
| **Zero setup** | Tinker | PrimeIntellect |

---

## Implementation Strategy

### Option A: Pure Compute (Recommended)

**Use PrimeIntellect as GPU provider, run existing code**

```bash
# 1. Deploy GPU
prime deploy --gpu a100 --count 1

# 2. Setup environment
ssh -i prime.pem ubuntu@<gpu-ip>
git clone https://github.com/you/trigoRL
cd trigoRL
pip install -e .

# 3. Train with existing code
python train_lm.py configs/training/trigo-gpt2.yaml

# 4. (Optional) Multi-node FSDP
prime deploy --gpu a100 --count 4 --cluster
torchrun --nnodes=4 train_distributed.py
```

**Effort**: 1 hour (deployment + setup)
**Cost**: $0.79/hr per A100
**Risk**: Low (no code changes)

**Verdict**: ✅ **Best option if you need more compute**

### Option B: Full prime-rl Integration

**Build custom Trigo environment for `prime-rl`**

**Phase 1: Environment Development (1-2 weeks)**
```python
# Create verifiers environment
class TrigoEnv(vf.MultiTurnEnv):
    def __init__(self, ...):
        # Initialize game engine
        ...

    async def env_response(self, messages, state):
        # Parse move from model text
        # Apply to game engine
        # Return next state
        ...
```

**Phase 2: Integration (3-5 days)**
```bash
# Configure prime-rl
cat > configs/trigo/rl.toml <<EOF
[trainer]
model_name = "gpt2"
...

[orchestrator]
env_id = "your-username/trigor-trigo"
...

[inference]
backend = "vllm"
...
EOF

# Launch training
uv run rl @ configs/trigo/rl.toml
```

**Phase 3: Testing (1 week)**
- Validate environment correctness
- Tune hyperparameters
- Scale to multi-GPU

**Total Effort**: 3-4 weeks
**Cost**: Development time + $0.79/hr GPU
**Risk**: Medium (integration complexity)

**Verdict**: ⚠️ **Only if you want async RL + community sharing**

### Option C: Hybrid Approach

**Use PrimeIntellect for compute, keep existing TrigoRL code**

**Phase 1: Port to PrimeIntellect (1 week)**
```bash
# 1. Package TrigoRL as Docker image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
COPY trigoRL /workspace/trigoRL
RUN pip install -e /workspace/trigoRL
...

# 2. Deploy on PrimeIntellect
prime deploy --image your-trigorl-image --gpu a100
```

**Phase 2: Add Simple RL (2 weeks)**
```python
# Finish trigor/training/rl_trainer.py
class RLTrainer:
    def train_episode(self):
        # Self-play with MCTS (existing C++ code)
        # Collect trajectories
        # PPO update
        ...
```

**Total Effort**: 3 weeks
**Cost**: $0.79/hr GPU
**Risk**: Low (builds on existing code)

**Verdict**: ✅ **Good balance of scalability and control**

---

## Final Recommendation

### For TrigoRL: Use PrimeIntellect for Compute Only

**Recommended Approach**:
1. ✅ **Start with Option A** (Pure Compute)
   - Deploy GPUs when needed
   - Run existing training code
   - Scale to multi-GPU with FSDP if needed

2. ⚠️ **Consider Option C** (Hybrid) if:
   - You finish `RLTrainer` implementation
   - Need scalable distributed training
   - Want to experiment with large models

3. ❌ **Skip Option B** (Full prime-rl) unless:
   - You want to contribute to Environments Hub
   - Need async off-policy RL specifically
   - Have time for 3-4 weeks of integration

### Decision Matrix

**Use PrimeIntellect if**:
- ✅ Need more GPU compute than you have locally
- ✅ Want to scale to multi-GPU without managing infrastructure
- ✅ Budget allows $50-200 for training experiments
- ✅ Want flexibility to try different cloud providers

**Stick with Local GPU if**:
- ✅ You already have sufficient GPU power
- ✅ Training time is acceptable
- ✅ Want zero cloud costs
- ✅ Prefer full control over hardware

**Don't use PrimeIntellect if**:
- ❌ You have free access to powerful local GPUs
- ❌ Don't need multi-GPU scaling
- ❌ Budget is very tight (<$50)

---

## Cost-Benefit Summary

### Quantitative Analysis

**Scenario**: Train 8B model for TrigoRL

| Metric | Local (RTX 4090) | PrimeIntellect (A100) | AWS (A100) |
|--------|------------------|----------------------|------------|
| **Hardware** | Free (already owned) | Rent | Rent |
| **Training Speed** | 1x | 3x | 3x |
| **Wall-Clock Time** | 150 hours | 50 hours | 50 hours |
| **Cost** | $0 | $39.50 | $150 |
| **Setup Time** | 0 | 5 minutes | 30 minutes |
| **Flexibility** | Full | Full | Full |

**Winner**: Local if you have it, PrimeIntellect if you need to rent

### Qualitative Analysis

**Advantages**:
- ✅ Much cheaper than Tinker for RL workloads (98% savings)
- ✅ Comparable to spot instances, more reliable
- ✅ Full control (SSH access, any code)
- ✅ Can build custom environments
- ✅ Multi-cloud aggregation
- ✅ No vendor lock-in

**Disadvantages**:
- ⚠️ Still need to manage training infrastructure
- ⚠️ `prime-rl` requires adaptation for games
- ⚠️ Per-hour charging (need to monitor usage)
- ⚠️ No built-in MCTS support

**Overall**: Much better fit than Tinker, but still requires work for full RL integration

---

## Conclusions

### Key Findings

1. **PrimeIntellect ≠ Tinker**: It's a compute platform, not a training API
2. **Better for Games**: More flexible than Tinker, supports custom code
3. **Cost Effective**: 50-80% cheaper than major clouds
4. **Two Use Modes**:
   - **Easy**: Just GPU rental (5 minutes setup)
   - **Advanced**: Full RL with custom environments (3-4 weeks)

### Recommendation Tiers

**Tier 1 - Highly Recommended**:
- ✅ Use PrimeIntellect for **GPU rental only**
- ✅ Run your existing TrigoRL code
- ✅ Scale to multi-GPU when needed

**Tier 2 - Conditionally Recommended**:
- ⚠️ Build custom Trigo environment for Environments Hub
- ⚠️ Only if you want community sharing
- ⚠️ Or if you need async off-policy RL specifically

**Tier 3 - Not Recommended**:
- ❌ Don't use `prime-rl` for AlphaZero-style training
- ❌ Don't use if you have sufficient local compute

### Next Steps

**If You Choose PrimeIntellect**:
1. Sign up at https://app.primeintellect.ai
2. Deploy 1x A100 for testing
3. SSH and clone TrigoRL
4. Run training with existing configs
5. Monitor costs and scale as needed

**If You Skip PrimeIntellect**:
1. Continue with local GPU training
2. Complete `RLTrainer` implementation
3. Integrate C++ MCTS
4. Revisit PrimeIntellect if you need more compute

---

## References

### PrimeIntellect Resources
- [Main Platform](https://app.primeintellect.ai/)
- [Documentation](https://docs.primeintellect.ai/)
- [Pricing Comparison](https://www.primeintellect.ai/competitor)
- [GitHub - prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
- [GitHub - verifiers](https://github.com/willccbb/verifiers)
- [Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- [Discord Community](https://discord.gg/ZTFydGWPKj)

### TrigoRL Resources
- [Project README](/home/camus/work/trigoRL/README.md)
- [CLAUDE.md](/home/camus/work/trigoRL/CLAUDE.md)
- [RL Trainer](/home/camus/work/trigoRL/trigor/training/rl_trainer.py)

### Related Analysis
- [Tinker Analysis](/home/camus/work/trigoRL/docs/tinker_analysis_for_trigorl.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-21
**Reviewer**: Claude (Sonnet 4.5)
