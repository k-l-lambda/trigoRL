# TransZero Research Summary and Insights for Trigo

**Date**: 2026-01-15
**Paper**: TransZero: Parallel Tree Expansion in MuZero using Transformer Networks (arXiv:2509.11233v1)
**Codebase**: https://github.com/emalmsten/TransZero

---

## Executive Summary

TransZero addresses a fundamental bottleneck in MuZero: **the sequential nature of MCTS tree construction**. By replacing MuZero's recurrent dynamics network with a **transformer-based parallel dynamics model**, combined with a **visit-count-free evaluator (MVC)**, TransZero achieves:

- ✅ **11× speedup** in wall-clock time (LunarLander)
- ✅ **2.5× speedup** (MiniGrid)
- ✅ **Same sample efficiency** as MuZero

**Key Innovation**: Remove sequential dependencies in both:
1. **Dynamics model**: Transformer generates entire action sequences in parallel
2. **Tree evaluation**: MVC evaluator eliminates dependence on visitation counts

---

### 🎯 CRITICAL DISCOVERY: Trigo Already Has Tree Attention Infrastructure!

**After analyzing Trigo's codebase**, we discovered that:

- ✅ **Tree-structured attention is already implemented** in `trigoTreeAgent.ts` and `prefix_tree_builder.hpp`
- ✅ **Ancestor attention masks are already working** for move evaluation
- ✅ **Batch processing with custom masks is already functional**

**Implication**: We can **reuse this infrastructure** for TransZero's parallel dynamics! This reduces:
- **Implementation complexity**: ~40% reduction
- **Development time**: From 6-10 weeks → **4-7 weeks**
- **Risk**: Much lower (proven code vs new implementation)

**Key insight**: Trigo uses tree attention for **evaluation** (scoring moves), TransZero uses it for **generation** (predicting states). Same mechanism, different application!

---

## 1. Core Problem: MuZero's Sequential Bottleneck

### 1.1 Sequential Dependencies in MuZero

```
Traditional MCTS in MuZero:

Simulation 1:
  Select → Expand → Backup
           ↓ (update visit counts)
Simulation 2:  ← must wait for Simulation 1
  Select → Expand → Backup
           ↓
Simulation 3:  ← must wait for Simulation 2
  ...
```

**Two sources of sequentiality**:

#### Problem 1: Recurrent Dynamics
```python
# MuZero dynamics (sequential)
s_1 = g(s_0, a_1)  # Step 1
s_2 = g(s_1, a_2)  # Step 2 (depends on s_1)
s_3 = g(s_2, a_3)  # Step 3 (depends on s_2)
...
```

Each state prediction requires the previous state → **K sequential forward passes**.

#### Problem 2: Visit Count Dependencies
```python
# PUCT selection (MuZero)
PUCT(s, a) = Q(s, a) + C * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
                                            ↑           ↑
                                    parent visits   child visits
```

Visit counts (N) are updated after each simulation → **selection depends on backup completion** → **inherently sequential**.

---

## 2. TransZero's Solutions

### 2.1 Transformer-Based Parallel Dynamics

Replace recurrent dynamics `g(s, a)` with transformer dynamics `g_trans`:

```python
# TransZero dynamics (parallel)
[s_1, s_2, s_3, ..., s_K] = g_trans(s_0, [a_1, a_2, a_3, ..., a_K])
                            ↑
                    Single forward pass!
```

**Architecture**:
```
Input:
  [s_root, embed(a_1), embed(a_2), ..., embed(a_K)]
     ↓
  Add positional encodings (depth in tree)
     ↓
  Transformer Encoder with causal mask
     ↓
Output:
  [s_0, s_1, s_2, ..., s_K]  (all latent states in parallel)
```

**Key Component: Tree-Structured Causal Mask**

```python
# Not standard causal mask (for sequences)
# But tree-structured mask (for MCTS tree)

Example tree:
         s_root
        /   |   \
      a_1  a_2  a_3
      /    /\    \
   a_11  a_21 a_22  a_31

Mask M_tree[i, j] = 1 if action_j is ancestor of action_i
                  = 0 otherwise

# Action a_22 can attend to: s_root, a_2 (ancestors)
# Action a_22 CANNOT attend to: a_21 (sibling), a_1, a_3 (uncle branches)
```

This allows **parallel expansion of entire subtrees** while maintaining correct information flow.

### 2.2 MVC Evaluator: Visit-Count-Free Evaluation

**Problem**: PUCT relies on visit counts → sequential updates.

**Solution**: Replace visit counts with **variance-based evaluation**.

#### Mean-Variance Constrained (MVC) Policy

```python
# MVC tree policy (evaluates nodes without visit counts)
π_MVC(x, a) ∝ π_Var(x, a) * exp(β * Q(x⊎a))

where:
  π_Var(x, a) = 1 / Variance[Q(x⊎a)]  # Prefer low variance (high certainty)
  β: balance parameter (0 → min variance, ∞ → max value)
```

**Intuition**:
- High variance = uncertain estimate → explore more
- Low variance = confident estimate → exploit if Q is high
- **No visit counts needed** → enables parallel expansion

#### Recursive Q and Variance Computation

```python
# Q-value (recursive)
Q(x) = r(x) + γ * Σ_a π_MVC(x, a) * Q(x⊎a)

# Variance (recursive)
V[Q(x)] = V[r(x)] + γ² * (π^T π) V[Q(children)]

# For leaf nodes:
V[Q(leaf)] = V[r(leaf)] + γ² * V[v(leaf)]
           = 0 + γ² * 1  (deterministic env, network variance = 1)
```

**PUCT Replacement**:
```python
# MuZero PUCT
U(x⊎a) = C * P(x,a) * sqrt(N(x)) / (1 + N(x⊎a))

# TransZero PUCT (visit-count-free)
U(x⊎a) = C * P(x,a) * sqrt(1/V[Q(x)]) / (1 + 1/V[Q(x⊎a)])
```

Variance replaces visit counts as a measure of certainty.

---

## 3. Parallel Subtree Expansion Algorithm

### 3.1 Overview

```
Traditional MuZero: Expand 1 node per simulation

TransZero: Expand entire subtree per simulation

Example (N_layers = 2):
         root
        /  |  \
      [Expand all 3 children in parallel]
      /   |   \
     c1  c2   c3
    / |  / \  | \
  [Expand all 7 grandchildren in parallel]
```

### 3.2 Detailed Process

**Step 1: Selection**
```python
# Use PUCT (with MVC) to select subtree root x*
x_star = argmax_child(PUCT_MVC(root, child))
```

**Step 2: Parallel Expansion**
```python
# Collect all actions in subtree (breadth-first)
action_sequence = [
    actions_to_x_star,  # Path from root to x_star
    all_actions_in_subtree(x_star, N_layers)
]

# Embed actions with positional encoding (tree depth)
action_embeds = [embed(a) + pos_encoding(depth(a)) for a in action_sequence]

# Generate all latent states in one pass
latent_states = transformer_dynamics(
    s_root,
    action_embeds,
    mask=M_tree  # Tree-structured mask
)

# Predict policy, value, reward for all states (batch)
policy, value, reward = prediction_head(latent_states)
```

**Step 3: Parallel Backup**
```python
# Compute Q and Variance bottom-up (by depth level)
for depth in range(N_layers, 0, -1):
    nodes_at_depth = get_nodes_at_depth(depth)

    # Parallel computation for all nodes at same depth
    for node in nodes_at_depth (parallel):
        Q(node) = r(node) + γ * Σ π_MVC * Q(children)
        V[Q(node)] = V[r] + γ² * π^T π * V[Q(children)]
```

**Key Insight**: Nodes at the same depth are independent → compute in parallel.

---

## 4. Architectural Details

### 4.1 Network Components

```python
class TransZeroNetwork(nn.Module):
    def __init__(self):
        # 1. Representation Network (same as MuZero)
        self.representation = ResNet(...)  # or CNN, MLP

        # 2. Action Embedding
        self.action_embedding = nn.Embedding(action_space, hidden_dim)

        # 3. Positional Encoding (depth in tree)
        self.pos_encoding = SinusoidalEncoding(max_depth, hidden_dim)
        # or: self.pos_encoding = nn.Embedding(max_depth, hidden_dim)

        # 4. Transformer Dynamics (KEY INNOVATION)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

        # 5. Prediction Heads (same as MuZero)
        self.policy_head = nn.Linear(hidden_dim, action_space)
        self.value_head = nn.Linear(hidden_dim, support_size)
        self.reward_head = nn.Linear(hidden_dim, support_size)

    def initial_inference(self, observation):
        """Encode root state"""
        s_root = self.representation(observation)
        policy, value = self.prediction_heads(s_root)
        return s_root, policy, value

    def recurrent_inference(self, s_root, action_sequence, tree_mask):
        """Parallel dynamics (KEY DIFFERENCE from MuZero)"""
        # Embed actions
        action_embeds = self.action_embedding(action_sequence)

        # Add positional encoding (tree depth)
        depths = compute_depths_from_tree(action_sequence)
        pos_embeds = self.pos_encoding(depths)
        action_embeds = action_embeds + pos_embeds

        # Concatenate root state and action embeds
        input_seq = torch.cat([s_root.unsqueeze(1), action_embeds], dim=1)

        # Transformer forward (with tree mask)
        latent_states = self.transformer(
            input_seq,
            mask=tree_mask  # Enforce tree structure
        )

        # Predict for all states in batch
        policy = self.policy_head(latent_states)
        value = self.value_head(latent_states)
        reward = self.reward_head(latent_states)

        return latent_states, policy, value, reward
```

### 4.2 Tree-Structured Masking

```python
def create_tree_mask(action_sequence, tree_structure):
    """
    Create attention mask that enforces tree structure.

    Args:
        action_sequence: [a_1, a_2, ..., a_K]
        tree_structure: parent indices for each action

    Returns:
        mask: [K+1, K+1] (including root)
    """
    K = len(action_sequence)
    mask = torch.zeros(K+1, K+1)

    # Root can attend to itself only
    mask[0, 0] = 1

    # Each action can attend to itself and ancestors
    for i, action in enumerate(action_sequence):
        action_idx = i + 1  # +1 for root

        # Allow attention to root
        mask[action_idx, 0] = 1

        # Allow attention to ancestors (follow parent pointers)
        current = action
        while current is not None:
            parent_idx = tree_structure[current]
            mask[action_idx, parent_idx + 1] = 1
            current = tree_structure[parent_idx]

        # Allow attention to self
        mask[action_idx, action_idx] = 1

    # Convert to attention mask format (0 → allowed, -inf → blocked)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)

    return mask
```

---

## 5. Key Insights for Trigo

### 5.1 Trigo's Existing Tree Attention Implementation

**IMPORTANT**: Trigo already implements tree-structured attention for move evaluation!

#### 5.1.1 Current Implementation

Located in:
- TypeScript: `/home/camus/work/trigo/trigo-web/inc/trigoTreeAgent.ts`
- C++ Header: `/home/camus/work/trigo.cpp/include/prefix_tree_builder.hpp`

**What it does** (lines 80-175 in trigoTreeAgent.ts):

```typescript
// Build prefix tree from move token sequences
private buildPrefixTree(tokenArrays: number[][]): {
    evaluatedIds: number[];      // Flattened token array
    mask: number[];              // Ancestor attention mask (m×m)
    moveToLeafPos: number[];     // Move-to-position mapping
    parent: Array<number | null>;
}
```

**Algorithm**:
1. **Recursive grouping**: Merge branches with the same token at EVERY level
2. **Flatten tree**: DFS traversal to create linear token sequence
3. **Ancestor mask**: Each node can attend to itself and all ancestors
   ```typescript
   // Lines 164-172: Build ancestor mask
   for (let i = 0; i < total; i++) {
       let p = i;
       while (p !== null) {
           mask[i * total + p] = 1;
           p = parent[p]!;
       }
   }
   ```
4. **Batch evaluation**: Single inference call scores all valid moves

**Use case**: Given current game state, evaluate ALL legal moves in parallel by organizing them as a prefix tree (e.g., "aa", "ab", "ba" share common prefixes).

#### 5.1.2 Comparison: Trigo's Tree Attention vs TransZero's

| Aspect | **Trigo (Current)** | **TransZero** |
|--------|---------------------|---------------|
| **Purpose** | Move evaluation (scoring) | MCTS tree expansion (generation) |
| **Tree type** | Token-level prefix tree | MCTS node tree |
| **Shares** | Move notation prefixes | State trajectories |
| **Inference mode** | **Evaluation** (forward pass, non-generative) | **Autoregressive generation** (recurrent dynamics) |
| **Time dimension** | Single step (t → score all actions at t) | Multi-step (t → t+k sequential states) |
| **Attention mask** | ✅ Ancestor attention | ✅ Ancestor attention (same concept!) |
| **Batch processing** | ✅ All moves at once | ✅ All nodes at once |

**Key Difference**:
- **Trigo**: "Given current position, what's the best move?" (横向并行 - horizontal parallelism)
- **TransZero**: "Given current position and action sequence, what are the future states?" (纵向并行 - vertical parallelism)

#### 5.1.3 Example: Same Mask, Different Use

**Trigo's use case** (Evaluation):
```
Current board state: s_t
Valid moves: ["aa", "ab", "b0"]

Prefix tree:
    root
    / \
   a   b
  / \   \
 a   b   0

Mask: Ancestor attention
Output: Score for each move (log probabilities)
Purpose: Select best move via policy
```

**TransZero's use case** (Generation):
```
MCTS tree expansion:
         s_root
        /   |   \
      a_1  a_2  a_3
      /    /\    \
   a_11  a_21 a_22  a_31

Mask: Ancestor attention (same concept!)
Output: Latent states [s_1, s_11, s_2, s_21, s_22, s_3, s_31]
Purpose: Expand multiple nodes for planning
```

**Bottom line**: Trigo already has the **infrastructure** (tree builder, mask generator) but uses it for **evaluation**, not **generation**.

---

### 5.2 Relevance Assessment (Revised)

| TransZero Feature | Trigo Status | Applicability | Priority | Difficulty |
|-------------------|--------------|---------------|----------|------------|
| **Tree-structured masking** | ✅ **Already implemented** | Reuse existing code | ~~High~~ **Low** | ~~Medium~~ **Low** |
| **Batch processing** | ✅ **Already implemented** | Reuse existing code | ~~High~~ **Low** | ~~Medium~~ **Low** |
| **Transformer dynamics** | ❌ **Missing** | High (for MCTS expansion) | **High** | Medium |
| **Parallel subtree expansion** | ❌ **Missing** | High (for MCTS speedup) | **High** | Medium |
| **MVC evaluator** | ❌ **Missing** | Medium (optional) | Low | Medium |

**Key Insight**: We DON'T need to implement tree masking from scratch! We can reuse `buildPrefixTree()` and adapt it for MCTS tree structure instead of token sequences.

---

### 5.3 Why These Features Matter for Trigo

#### 5.3.1 Transformer Dynamics (High Priority - Missing)

**Current Trigo**:
- Already uses Transformer architecture
- But only for **encoding move sequences**
- Dynamics still rely on **game rules** (not learned)
- Tree attention used only for **evaluation**, not **generation**

**What TransZero Offers**:
- **Learned dynamics in latent space**
- Can generate multiple future states in parallel
- Faster MCTS tree expansion
- **Reuses existing tree mask infrastructure** for dynamics generation

**Potential Benefit**:
```python
# Current Trigo MCTS (50 simulations)
for sim in range(50):
    node = select(root)
    next_board = trigo_game.make_move(board, action)  # Game rules
    value = model.evaluate(next_board)
    backup(node, value)

# With TransZero-style (parallel)
# Generate entire tree in 1-2 forward passes
action_sequences = generate_tree_actions(root, depth=3)  # 3^depth nodes

# REUSE existing buildPrefixTree() for MCTS nodes!
tree_structure = buildMCTSTree(action_sequences)  # Adapted from trigoTreeAgent

latent_states = model.transformer_dynamics(
    root_latent,
    action_sequences,
    mask=tree_structure.mask  # Reuse existing mask generator!
)
values = model.value_head(latent_states)  # Batch evaluation
parallel_backup(tree, values)
```

**Expected Speedup**: 3-5× for MCTS-heavy training.

#### 5.3.2 Parallel Subtree Expansion (High Priority - Missing)

**Current Trigo MCTS**: Sequential node expansion.

**With Parallel Expansion**:
- Expand 3^2 = 9 nodes (depth 2) in one shot
- Especially valuable for:
  - Long MCTS rollouts (>50 simulations)
  - Real-time gameplay requirements
  - Distributed training (reduce communication overhead)

**Advantage**: Can **reuse trigoTreeAgent's tree building logic** by adapting it from token sequences to MCTS nodes!

**Challenge for Trigo**:
- Trigo's action space is small (26 positions + pass)
- But branching factor is high (average ~20 legal moves)
- Depth-2 subtree = 20×20 = 400 nodes → may exceed memory

**Solution**: Adaptive subtree depth based on branching factor.

#### 5.3.3 MVC Evaluator (Lower Priority - Missing)

**Why Lower Priority**:
- Trigo's games are relatively short (30 steps average)
- Visit counts provide useful exploration signal
- MVC adds complexity without clear benefit

**Potential Use Case**:
- If we implement parallel expansion, MVC becomes necessary
- Otherwise, visit counts work well

#### 5.3.4 Tree-Structured Masking (Low Priority - Already Implemented!)

~~**Critical for Correctness**~~:
**GOOD NEWS**: Already implemented in `trigoTreeAgent.buildPrefixTree()`!

**What we need to do**:
- ✅ Mask generation algorithm: Already implemented
- ✅ Ancestor attention pattern: Already working
- ⚠️ Adaptation needed: Change input from token sequences to MCTS node sequences

**Code reuse strategy**:
```typescript
// Current (trigoTreeAgent.ts, line 80):
buildPrefixTree(tokenArrays: number[][]): TreeStructure

// New (for TransZero):
buildMCTSTree(mctsNodeSequences: MCTSNode[][]): TreeStructure
// Same algorithm, different input type!
```

**Trigo-Specific Consideration**:
- Trigo's move sequence is linear (no branching in game history)
- But MCTS tree has branches → need tree mask for search
- **We already have this for move evaluation** → just adapt for dynamics generation

---

## 6. Implementation Strategy for Trigo (REVISED)

### 6.1 Key Advantage: Existing Infrastructure

**GOOD NEWS**: Trigo already has most of the tree attention infrastructure implemented in `trigoTreeAgent.ts` and `prefix_tree_builder.hpp`!

**What we can reuse**:
- ✅ Tree building algorithm (recursive grouping, flattening)
- ✅ Ancestor mask generation
- ✅ Batch processing with custom masks
- ✅ Parent array tracking for path reconstruction

**What we need to add**:
- ❌ Transformer-based dynamics head (for state generation, not just evaluation)
- ❌ MCTS-specific tree builder (adapt from token sequences to node sequences)
- ❌ Parallel backup algorithm (MVC or visit-count-based)

**Estimated reduction in implementation time**: 30-40% (tree masking infrastructure already exists)

---

### 6.2 Phased Approach (Revised)

#### Phase 1: Adapt Tree Builder for MCTS (1 week)

**Goal**: Reuse `buildPrefixTree()` for MCTS node sequences instead of token sequences.

**Steps**:
1. Create `buildMCTSTree()` function that wraps `buildPrefixTree()`
   - Input: MCTS node sequences (action paths)
   - Output: Tree structure with mask (same format as trigoTreeAgent)
2. Unit tests: Verify mask correctness for MCTS trees
3. Visualize attention patterns for debugging

**Success Criteria**: Generate correct ancestor masks for MCTS subtrees.

**Code adaptation**:
```python
# Wrapper around existing buildPrefixTree logic
def build_mcts_tree(mcts_root, depth):
    """
    Adapt trigoTreeAgent.buildPrefixTree() for MCTS nodes.

    Args:
        mcts_root: Root node of MCTS subtree
        depth: Number of layers to expand

    Returns:
        TreeStructure with mask (same format as trigoTreeAgent)
    """
    # Collect action sequences (BFS)
    action_sequences = collect_action_paths(mcts_root, depth)

    # Reuse existing algorithm!
    # (Port TypeScript logic or call via bridge)
    tree_structure = build_prefix_tree(action_sequences)

    return tree_structure
```

#### Phase 2: Transformer Dynamics (1-2 weeks)

**Goal**: Learn dynamics model in Trigo's latent space.

**Steps**:
1. Add transformer-based dynamics head
2. **Integrate with existing tree mask infrastructure**
3. Train to predict next latent state given (state, action)
4. Validate against true game rules

**Success Criteria**: Dynamics prediction accuracy >95%.

**Code Structure** (reusing tree mask):
```python
class TrigoTransformerDynamics(nn.Module):
    def __init__(self, base_model):
        self.base_model = base_model  # Existing Llama

        # Transformer dynamics
        self.action_embedding = nn.Embedding(26, hidden_size)
        self.pos_encoding = SinusoidalEncoding(max_seq_len, hidden_size)

        self.dynamics_transformer = nn.TransformerEncoder(...)
        self.dynamics_head = nn.Linear(hidden_size, hidden_size)

    def predict_next_latent(self, latent, action_sequence, tree_mask):
        """
        Args:
            latent: [batch, hidden] current state
            action_sequence: [batch, K] next K actions
            tree_mask: [K, K] from buildMCTSTree() (REUSE!)

        Returns:
            next_latents: [batch, K, hidden]
        """
        # Embed actions
        action_embeds = self.action_embedding(action_sequence)
        action_embeds = action_embeds + self.pos_encoding(range(K))

        # Concatenate with latent
        input_seq = torch.cat([latent.unsqueeze(1), action_embeds], dim=1)

        # Transformer forward with TREE MASK (reused from trigoTreeAgent!)
        output = self.dynamics_transformer(input_seq, mask=tree_mask)

        # Dynamics head
        next_latents = self.dynamics_head(output[:, 1:, :])

        return next_latents
```

#### Phase 3: Basic Parallel Expansion (1-2 weeks)

**Goal**: Expand multiple nodes simultaneously.

**Steps**:
1. ✅ Tree mask generation (already done in Phase 1!)
2. Modify MCTS to expand small subtrees (depth=2)
3. Implement parallel backup (visit-count-based first)
4. Measure speedup vs sequential

**Success Criteria**: 2× speedup in MCTS.

#### Phase 4: MVC Integration (1-2 weeks, Optional)

**Goal**: Replace visit counts with variance-based evaluation.

**Steps**:
1. Implement recursive Q/Variance computation
2. Modify PUCT to use variance instead of visit counts
3. A/B test against standard MCTS

**Success Criteria**: Maintain performance while enabling parallelism.

---

### 6.3 Timeline Summary (Revised)

| Phase | Original Estimate | Revised Estimate | Reason |
|-------|------------------|------------------|--------|
| Tree masking | 1-2 weeks | ✅ **Done** | Already implemented! |
| Adapt for MCTS | N/A | 1 week | Simple wrapper |
| Transformer dynamics | 2-3 weeks | 1-2 weeks | Reuse tree infrastructure |
| Parallel expansion | 2-3 weeks | 1-2 weeks | Tree building already done |
| MVC (optional) | 1-2 weeks | 1-2 weeks | No change |
| **Total (Phases 1-3)** | **5-8 weeks** | **3-5 weeks** | **~40% faster** |
| **Total (all phases)** | **6-10 weeks** | **4-7 weeks** | **~40% faster** |

---

## 7. Expected Benefits and Risks

### 7.1 Expected Benefits

| Benefit | Magnitude | Confidence |
|---------|-----------|------------|
| **MCTS Speedup** | 2-5× | High |
| **Training Speedup** | 1.5-2× | Medium |
| **Sample Efficiency** | No change | High |
| **Scalability** | Better for large trees | Medium |

### 7.2 Risks and Mitigations

#### Risk 1: Dynamics Prediction Accuracy

**Issue**: Learned dynamics may be less accurate than true rules.

**Mitigation**:
- Start with hybrid approach: use true rules for root expansions, learned dynamics for deep simulations
- Monitor prediction accuracy during training
- Use dynamics only when accuracy >95%

#### Risk 2: Memory Overhead

**Issue**: Parallel expansion requires storing entire subtrees.

**Mitigation**:
- Adaptive depth: use depth=1 for high branching factor, depth=2-3 for low
- Gradient checkpointing for large trees
- Limit subtree size to 100 nodes

#### Risk 3: Implementation Complexity

**Issue**: Tree masking and parallel backup add complexity.

**Mitigation**:
- Start with small subtrees (depth=2, ~9 nodes)
- Thorough unit tests for mask generation
- Visualize attention patterns to debug

---

## 8. Key Differences: TransZero vs UniZero vs Trigo

| Feature | **TransZero** | **UniZero** | **Trigo (Current)** |
|---------|--------------|------------|---------------------|
| **Primary Innovation** | Parallel MCTS | Disentangle state/history | Transformer for Go |
| **Dynamics Model** | Transformer (parallel) | Transformer (sequential) | Game rules |
| **Latent State** | From dynamics | From encoder | KV cache |
| **MCTS Parallelism** | ✅ Entire subtrees | ❌ Sequential | ❌ Sequential |
| **Visit Count Free** | ✅ MVC evaluator | ❌ Uses counts | ❌ Uses counts |
| **Self-Supervised Loss** | ❌ Not mentioned | ✅ Latent prediction | ❌ Not yet |
| **Main Goal** | Speedup | Sample efficiency | Game playing |

**Synergy**: TransZero + UniZero could be combined:
- UniZero's self-supervised latent prediction
- TransZero's parallel tree expansion
- Best of both worlds!

---

## 9. Code Examples

### 9.1 Tree Mask Generation

```python
def create_tree_mask_for_mcts(tree, root_node, depth):
    """
    Generate tree-structured attention mask for MCTS subtree.

    Args:
        tree: MCTS tree structure
        root_node: Root of subtree to expand
        depth: Number of layers to expand

    Returns:
        mask: [num_nodes, num_nodes] attention mask
        node_list: List of nodes in breadth-first order
    """
    # Collect all nodes in subtree (breadth-first)
    node_list = [root_node]
    queue = [root_node]

    for d in range(depth):
        next_level = []
        for node in queue:
            for action in node.legal_actions():
                child = node.children.get(action)
                if child is None:
                    child = MCTSNode(parent=node, action=action)
                    node.children[action] = child
                node_list.append(child)
                next_level.append(child)
        queue = next_level

    # Build parent index mapping
    num_nodes = len(node_list)
    parent_indices = {}
    for i, node in enumerate(node_list):
        parent_indices[node] = node_list.index(node.parent) if node.parent else -1

    # Create mask
    mask = torch.zeros(num_nodes, num_nodes)
    for i, node in enumerate(node_list):
        # Allow attention to self
        mask[i, i] = 1

        # Allow attention to ancestors
        current = node.parent
        while current is not None and current in node_list:
            j = node_list.index(current)
            mask[i, j] = 1
            current = current.parent

    # Convert to attention mask format
    mask = (1 - mask) * -1e9

    return mask, node_list
```

### 9.2 Parallel Dynamics Forward

```python
class TrigoParallelDynamics(nn.Module):
    def forward(self, root_latent, action_tree, tree_mask):
        """
        Args:
            root_latent: [1, hidden] root state embedding
            action_tree: [num_nodes-1] actions in tree (BFS order)
            tree_mask: [num_nodes, num_nodes] tree attention mask

        Returns:
            latent_states: [num_nodes, hidden] all node embeddings
        """
        # Embed actions
        action_embeds = self.action_embedding(action_tree)  # [num_nodes-1, hidden]

        # Add positional encoding (depth in tree)
        depths = compute_depths_from_actions(action_tree)
        pos_embeds = self.pos_encoding(depths)
        action_embeds = action_embeds + pos_embeds

        # Concatenate root and actions
        input_seq = torch.cat([
            root_latent.unsqueeze(0),  # [1, hidden]
            action_embeds               # [num_nodes-1, hidden]
        ], dim=0)  # [num_nodes, hidden]

        # Transformer forward with tree mask
        latent_states = self.transformer(
            input_seq.unsqueeze(0),  # Add batch dim
            mask=tree_mask
        ).squeeze(0)  # [num_nodes, hidden]

        return latent_states
```

### 9.3 Parallel Backup

```python
def parallel_backup(tree, node_list, values):
    """
    Backup values through tree in parallel (level by level).

    Args:
        tree: MCTS tree
        node_list: List of nodes (BFS order)
        values: [num_nodes] predicted values
    """
    # Assign values to nodes
    for node, value in zip(node_list, values):
        node.predicted_value = value

    # Compute Q and Variance level by level (bottom-up)
    max_depth = max(node.depth for node in node_list)

    for depth in range(max_depth, -1, -1):
        nodes_at_depth = [n for n in node_list if n.depth == depth]

        # Parallel computation for all nodes at same depth
        for node in nodes_at_depth:
            if node.is_leaf():
                node.Q = node.predicted_value
                node.V = 1.0  # Variance for leaf
            else:
                # Compute MVC policy
                pi_mvc = compute_mvc_policy(node, beta=1.0)

                # Recursive Q
                node.Q = node.reward + gamma * sum(
                    pi_mvc[a] * node.children[a].Q
                    for a in node.children
                )

                # Recursive Variance
                child_V = [node.children[a].V for a in node.children]
                node.V = gamma**2 * (pi_mvc @ pi_mvc) @ child_V
```

---

## 10. Conclusion (REVISED)

TransZero's core innovation—**parallel tree expansion through transformer dynamics**—is highly relevant to Trigo. **Crucially, Trigo has already implemented the tree-structured attention infrastructure** (`trigoTreeAgent.ts`, `prefix_tree_builder.hpp`) for move evaluation, which can be **directly reused** for MCTS dynamics generation.

### 10.1 Key Discovery: Existing Infrastructure

**What we already have**:
- ✅ Tree building algorithm (recursive grouping by prefix)
- ✅ Ancestor attention mask generation
- ✅ Batch processing with custom masks
- ✅ Parent array tracking for path reconstruction

**What we need to add** (much simpler than implementing from scratch):
- ❌ Adapt tree builder from token sequences → MCTS node sequences
- ❌ Transformer dynamics head (learns state transitions)
- ❌ Parallel backup algorithm

**Impact**: Reduces implementation complexity by ~40% and time from **6-10 weeks** to **4-7 weeks**.

---

### 10.2 Revised Next Steps

**Phase 0 (COMPLETED)**: Tree attention infrastructure
   - ✅ Already implemented in `trigoTreeAgent.ts`
   - ✅ Used for parallel move evaluation
   - ✅ Ancestor mask generation working

**Phase 1 (1 week)**: Adapt tree builder for MCTS
   - Wrap `buildPrefixTree()` to accept MCTS node sequences
   - Verify mask correctness for MCTS subtrees
   - Unit tests and visualization

**Phase 2 (1-2 weeks)**: Transformer dynamics
   - Add dynamics head to predict next latent states
   - **Integrate with existing tree mask** (from Phase 1)
   - Validate against true game rules
   - Measure prediction accuracy (target: >95%)

**Phase 3 (1-2 weeks)**: Parallel subtree expansion
   - Modify MCTS to expand small subtrees (depth=2)
   - Implement parallel backup (visit-count-based)
   - Benchmark speedup (target: 2× minimum)

**Phase 4 (1-2 weeks, Optional)**: MVC evaluator
   - Replace visit counts with variance
   - A/B test vs standard MCTS

**Total timeline**: 3-5 weeks (core features), 4-7 weeks (including optional MVC)

---

### 10.3 Strategic Insight: TransZero + UniZero

TransZero and UniZero are **complementary**:
- **UniZero**: Better sample efficiency (self-supervised latent learning)
- **TransZero**: Better wall-clock efficiency (parallel MCTS)
- **Both share**: Transformer backbone and tree-structured planning

**Synergy for Trigo**:
1. Use UniZero's latent dynamics prediction head
2. Use TransZero's parallel tree expansion algorithm
3. Use **Trigo's existing tree attention infrastructure** for both!

**Expected combined benefits**:
- 10-30% sample efficiency improvement (UniZero)
- 2-5× MCTS speedup (TransZero)
- 1.5-3× overall training acceleration

---

### 10.4 Bottom Line

**Before this analysis**: Thought we needed to implement tree attention from scratch (high complexity).

**After this analysis**: Discovered we already have it! Just need to:
1. Adapt for MCTS (1 week)
2. Add dynamics head (1-2 weeks)
3. Enable parallel expansion (1-2 weeks)

**Recommendation**: Start with Phase 1 (MCTS tree adapter) to validate the code reuse approach, then proceed to Phases 2-3 for core TransZero features.

---

## 11. References

### Papers
1. **TransZero** (Malmsten & Böhmer, 2024): https://arxiv.org/abs/2509.11233
2. **MuZero** (Schrittwieser et al., 2019): https://arxiv.org/abs/1911.08265
3. **UniZero** (Pu et al., 2024): https://arxiv.org/abs/2406.10667
4. **General Tree Evaluation** (Jaldevik, 2024): Framework for visit-count-free MCTS

### Code
1. **TransZero**: https://github.com/emalmsten/TransZero
   - Key files:
     - `src/trans_zero/networks/transformer.py`: Transformer dynamics
     - MCTS with parallel expansion

2. **LightZero** (UniZero): https://github.com/opendilab/LightZero
   - Reference for self-supervised latent learning

---

**Document Version**: 2.0 (Revised after discovering Trigo's existing tree attention implementation)
**Last Updated**: 2026-01-15
**Author**: Claude Code (based on TransZero paper, code analysis, and Trigo codebase review)

**Revision History**:
- v1.0 (2026-01-15): Initial analysis of TransZero architecture
- v2.0 (2026-01-15): Added analysis of Trigo's existing tree attention infrastructure, revised implementation strategy with 40% time reduction
