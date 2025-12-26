# CMS Dynamic Block Sparse Linear Layer Specification

**Version**: 1.0.0  
**Date**: 2024-12-25  
**Status**: Implementation Ready  
**Target Integration**: TitanMAC (237M parameter memory-augmented transformer)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [System Architecture](#3-system-architecture)
4. [Data Structures](#4-data-structures)
5. [Algorithms](#5-algorithms)
6. [Hardware Considerations](#6-hardware-considerations)
7. [Triton Implementation](#7-triton-implementation)
8. [CMS Integration](#8-cms-integration)
9. [Learned Topology Controller](#9-learned-topology-controller)
10. [TitanMAC Integration](#10-titanmac-integration)
11. [Testing Strategy](#11-testing-strategy)
12. [Success Metrics](#12-success-metrics)

---

## 1. Problem Statement

### 1.1 Catastrophic Forgetting

Neural networks suffer from catastrophic forgetting: when trained on Task B after Task A, performance on Task A degrades severely. The standard explanation is that gradient descent overwrites weights that encoded Task A knowledge with weights optimized for Task B.

**Current mitigations and their limitations:**

| Approach | Mechanism | Limitation |
|----------|-----------|------------|
| EWC (Elastic Weight Consolidation) | Penalize changes to "important" weights | Importance estimated on old task, not adaptive |
| Progressive Networks | Add new columns per task | Linear growth in parameters |
| Replay Buffers | Retrain on old data | Requires storing old data |
| PackNet | Prune and freeze per task | Fixed capacity allocation |

**Our hypothesis:** Dynamic topology enables knowledge preservation without explicit freezing. The network can:
1. **Preserve** existing pathways (connections) that encode Task A
2. **Grow** new pathways for Task B in unused capacity
3. **Route around** instead of overwrite

This is analogous to biological neural plasticity, where new synapses form while important existing synapses are maintained.

### 1.2 The Efficiency Problem

Standard "sparse" implementations don't actually speed up computation:

```python
# Typical sparse implementation (e.g., RigL)
weight = torch.randn(1024, 1024)  # Dense storage
mask = torch.zeros(1024, 1024)     # Binary mask
mask[sparse_indices] = 1
output = input @ (weight * mask)   # Still dense matmul!
```

This performs:
- **Dense memory load**: All 1M weights loaded from DRAM
- **Dense computation**: All 1M multiply-adds executed
- **Mask overhead**: Additional elementwise multiply

Result: 90% sparsity gives 0% speedup, often negative due to mask overhead.

**Why this happens:** GPUs are throughput machines optimized for regular memory access patterns. Scattered individual zeros don't help—the memory system still fetches full cache lines, and compute units still process full vectors.

### 1.3 Our Solution: Block-Sparse with Learned Topology

We combine three key ideas:

1. **Block sparsity**: Prune/grow entire tiles (16×16), not individual weights
   - Matches GPU memory hierarchy (128-byte cache lines)
   - Enables Tensor Core utilization (WMMA instructions)
   - Amortizes indexing overhead over 256 parameters

2. **Dynamic topology**: Connections evolve during training
   - Unlike static pruning (prune once, train, done)
   - Unlike unstructured dynamic sparsity (RigL—correct idea, wrong granularity)

3. **Learned topology decisions**: Neural network decides what to prune/grow
   - Not hand-designed heuristics (magnitude pruning, gradient growth)
   - Trained via meta-learning to optimize validation performance
   - Can learn task-specific routing strategies

---

## 2. Theoretical Foundation

### 2.1 Connection to Neuroevolution (NEAT)

NEAT (NeuroEvolution of Augmenting Topologies) demonstrated that evolving network structure alongside weights produces better solutions than fixed architectures. Key NEAT insights we inherit:

1. **Topology matters**: The pattern of connections affects what a network can learn
2. **Incremental complexification**: Start simple, add structure as needed
3. **Innovation protection**: New structures need time to optimize before judgment

NEAT operates on individual neurons/connections. We operate on **tiles** (blocks of neurons) for hardware efficiency. This is a coarser granularity but maintains the core insight: structure should adapt.

### 2.2 Lottery Ticket Hypothesis Connection

The Lottery Ticket Hypothesis (Frankle & Carlin, 2019) proved that dense networks contain sparse subnetworks ("winning tickets") that can match dense performance when trained in isolation.

**Key insight for us:** The winning tickets exist at initialization. Our dynamic topology is searching for these tickets during training, not just at the end.

The block granularity is supported by follow-up work showing that structured lottery tickets exist—you don't need individual weight sparsity to find winning subnetworks.

### 2.3 Multi-Timescale Learning

Different aspects of learning operate at different timescales:

| Timescale | What Changes | Frequency |
|-----------|--------------|-----------|
| Fast (steps) | Weight values | Every gradient step |
| Medium (10s of steps) | Importance estimates | Accumulated statistics |
| Slow (100s of steps) | Network structure | Topology decisions |

This mirrors biological learning:
- **Synaptic plasticity** (fast): Weight changes via LTP/LTD
- **Synaptic tagging** (medium): Marking synapses for consolidation
- **Structural plasticity** (slow): Synapse formation/elimination

TitanMAC already implements multi-timescale learning via CMS (Continuum Memory System). We add topology changes as another slow timescale.

### 2.4 Why Learned Scoring Beats Heuristics

Standard dynamic sparsity uses:
- **Prune criterion**: Weight magnitude (`|W|`)
- **Grow criterion**: Gradient magnitude (`|∇W|`)

These are proxy metrics, not direct optimization targets. Problems:

1. **Magnitude ≠ importance**: A large weight encoding memorized noise should be pruned; a small weight in a critical pathway should be kept.

2. **Gradient magnitude is batch-dependent**: High gradient on current batch doesn't mean the connection is globally useful.

3. **No adaptation**: The scoring function doesn't improve over training.

**Our approach:** Train a small MLP to predict "should this block be kept/added?" using:
- Accumulated gradient statistics (EMA over many steps)
- Block age (how long has this connection existed)
- Structural context (row density, column popularity)

The MLP is trained via meta-learning: topology decisions at step T are evaluated by validation loss at step T+k. This directly optimizes for generalization, not training fit.

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TitanMAC Model                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    TitanBlock (x16)                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │  Attention  │  │   MLP       │  │  Neural Memory  │   │  │
│  │  │  (dense)    │  │  (SPARSE)   │  │  (dense)        │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DeepNestedOptimizer                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  CMS (Continuum Memory System)            │  │
│  │  Level 0 (freq=1):   Weight updates                       │  │
│  │  Level 1 (freq=10):  Score accumulation ──────────────┐   │  │
│  │  Level 2 (freq=100): Topology decisions ◄─────────────┤   │  │
│  └───────────────────────────────────────────────────────┼───┘  │
│                                                          │      │
│  ┌───────────────────────────────────────────────────────┼───┐  │
│  │              Topology Controller (MLP)                │   │  │
│  │  Input: [score_ema, age, row_density, col_popularity] │   │  │
│  │  Output: keep/add score                               │   │  │
│  │  Training: Meta-learning on validation loss           ◄───┘  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

**CMSBlockLinear** (drop-in replacement for nn.Linear):
- Stores weights in Block-ELL format
- Executes block-sparse forward pass (Triton kernel)
- Accumulates per-block gradient statistics
- Exposes topology update interface

**TopologyController** (shared across layers):
- Scores existing blocks for pruning decisions
- Scores candidate blocks for growth decisions
- Trained by meta-optimizer alongside momentum MLP

**DeepNestedOptimizer** (existing, extended):
- Manages CMS state for all parameters
- Triggers Level 1 (score_step) and Level 2 (topology_step)
- Trains TopologyController via meta-learning

### 3.3 Data Flow

```
Forward Pass:
  input ──► CMSBlockLinear.forward() ──► output
              │
              ├── Track activation norms per input block
              └── Triton kernel: block-sparse matmul

Backward Pass:
  grad_output ──► CMSBlockLinear.backward() ──► grad_input
                    │
                    ├── Compute grad_values (per active block)
                    └── Track error norms per output block

After Backward (every step):
  CMSBlockLinear.accumulate_scores()
    └── Update block_score_ema from grad_values norms

Level 1 (every 10 steps):
  CMSBlockLinear.score_step()
    ├── Normalize accumulated activation/error norms
    ├── Increment block ages
    └── Reset step counter

Level 2 (every 100 steps):
  CMSBlockLinear.topology_step()
    ├── Gather features for existing blocks
    ├── Sample and score candidate blocks
    ├── TopologyController scores all blocks
    ├── Select top-K per row
    ├── Reallocate weights (copy survivors, init newcomers)
    └── Reset accumulators
```

---

## 4. Data Structures

### 4.1 Block-ELL Format

Block-ELL (Block ELLPACK) is a sparse matrix format optimized for fixed-fanout sparse matrices.

**Logical view** (what the layer represents):
```
Weight matrix W: [out_features, in_features]
                 [2560, 640] for fc1
```

**Physical storage** (how it's stored):
```
values:      [R, K, B, B]  = [160, 20, 16, 16]  # Actual weights
col_indices: [R, K]        = [160, 20]          # Which columns are active

where:
  R = out_features // B = 2560 // 16 = 160 (output block-rows)
  K = active blocks per row (hyperparameter, e.g., 20 for 50% density)
  B = block/tile size = 16
```

**Memory layout:**
```
values[r, k, i, j] is at offset: r * (K * B * B) + k * (B * B) + i * B + j

For r=0, k=0: values[0,0,:,:] occupies bytes 0-1023 (256 floats × 4 bytes)
For r=0, k=1: values[0,1,:,:] occupies bytes 1024-2047
...contiguous in memory
```

**Why this layout:**
1. All weights for one output block-row are contiguous
2. Loading values[r, k, :, :] is a single 1KB read (fits in L1 cache)
3. No pointer chasing for weights—only col_indices requires indirection

### 4.2 Index Semantics

```python
col_indices: Tensor[int32], shape [R, K]

# col_indices[r, k] = c means:
#   "The k-th active block in output row r connects to input column c"
#   "values[r, k, :, :] multiplies with input[*, c*B : (c+1)*B]"

# Sentinel value: -1 means "empty slot" (unused capacity)
# In practice, we keep all K slots filled, so -1 is only during init
```

**Visualization:**
```
Output Block-Row 0:
  col_indices[0] = [3, 7, 12, 25, ...]  (K=20 entries)
  
  This means output[0:16] = Σ values[0,k] @ input[col_indices[0,k]*16 : ...]
                          = values[0,0] @ input[48:64]    (col 3)
                          + values[0,1] @ input[112:128]  (col 7)
                          + values[0,2] @ input[192:208]  (col 12)
                          + ...
```

### 4.3 CMS State Buffers

Per-layer state for topology decisions:

```python
# Accumulated per-block importance (updated every backward pass)
block_score_ema: Tensor[float32], shape [R, K]
# Entry [r, k] = EMA of gradient Frobenius norm for block (r, k)

# Block age in topology steps (how long since birth/last swap)
block_age: Tensor[int32], shape [R, K]
# Entry [r, k] = number of Level 2 steps since this block was created

# Accumulated input activation norms (for candidate scoring)
activation_norm_acc: Tensor[float32], shape [C]
# Entry [c] = accumulated L2 norm of input block c across all steps

# Accumulated output error norms (for candidate scoring)  
error_norm_acc: Tensor[float32], shape [R]
# Entry [r] = accumulated L2 norm of gradient at output block r

# Step counter for normalization
acc_steps: Tensor[int64], scalar
# Number of steps since last Level 1 reset
```

### 4.4 Global State (Shared Across Layers)

```python
# Column usage counts (which input blocks are popular)
# Computed from col_indices across all layers
col_usage_global: Tensor[float32], shape [C]

# Topology controller (learned)
topology_controller: TopologyController
# Shared across all CMSBlockLinear layers
# Trained by meta-optimizer
```

---

## 5. Algorithms

### 5.1 Forward Pass

**Mathematical operation:**
```
For output block-row r (computing output[r*B : (r+1)*B]):

  y_r = Σ_{k=0}^{K-1} V[r,k] @ x[c_k]
  
  where c_k = col_indices[r, k]
        V[r,k] is B×B weight matrix
        x[c_k] is B-dimensional input slice
```

**Algorithm:**
```python
def forward(x: Tensor[batch, in_features]) -> Tensor[batch, out_features]:
    # Reshape input to blocked view
    x_blocked = x.view(batch, C, B)  # [batch, num_input_blocks, block_size]
    
    # Initialize output
    output = zeros(batch, R, B)
    
    # Accumulate contributions from each active block
    for k in range(K):
        col_idx = col_indices[:, k]  # [R] - one column index per row
        
        # Gather input blocks (the indirect access)
        # input_blocks[r] = x_blocked[:, col_idx[r], :]
        input_blocks = gather(x_blocked, dim=1, index=col_idx)  # [batch, R, B]
        
        # Batch matmul: values[:, k, :, :] @ input_blocks
        # [R, B, B] @ [batch, R, B] -> [batch, R, B]
        output += einsum('rij, nrj -> nri', values[:, k], input_blocks)
    
    return output.view(batch, out_features)
```

**Complexity:**
- Dense: O(batch × R × C × B²) = O(batch × out × in)
- Sparse: O(batch × R × K × B²) = O(batch × out × in × density)
- Speedup: 1/density (e.g., 2× at 50% density, 10× at 10% density)

### 5.2 Backward Pass

**Gradient computation:**

```
Given grad_output: [batch, out_features]

Need:
  grad_values[r, k, i, j] = ∂L/∂V[r,k,i,j]
  grad_input[*, c*B+j] = ∂L/∂x[c,j]
```

**Algorithm:**
```python
def backward(grad_output, x, values, col_indices):
    grad_out_blocked = grad_output.view(batch, R, B)
    x_blocked = x.view(batch, C, B)
    
    grad_values = zeros_like(values)
    grad_x_blocked = zeros(batch, C, B)
    
    for k in range(K):
        col_idx = col_indices[:, k]  # [R]
        
        # Gather input blocks for this slot
        input_blocks = gather(x_blocked, dim=1, index=col_idx)  # [batch, R, B]
        
        # grad_values[r,k] = Σ_batch grad_out[batch,r] ⊗ input[batch,c_r]
        # Outer product summed over batch
        grad_values[:, k] = einsum('nri, nrj -> rij', grad_out_blocked, input_blocks)
        
        # grad_x[c_r] += values[r,k]^T @ grad_out[r]
        # Scatter-add back to input gradient
        grad_contrib = einsum('rij, nri -> nrj', values[:, k], grad_out_blocked)
        scatter_add_(grad_x_blocked, dim=1, index=col_idx, src=grad_contrib)
    
    return grad_x_blocked.view(batch, in_features), grad_values
```

### 5.3 Score Accumulation (Every Step)

After each backward pass:

```python
def accumulate_scores(self):
    if self.values.grad is None:
        return
    
    # Frobenius norm of gradient for each block
    # High gradient norm = this block is being actively used for learning
    instant_scores = self.values.grad.pow(2).sum(dim=(2, 3)).sqrt()  # [R, K]
    
    # EMA update (smooth over steps to reduce batch noise)
    # α = 0.1 means ~10 step half-life
    self.block_score_ema = 0.9 * self.block_score_ema + 0.1 * instant_scores
```

**Why gradient norm, not weight magnitude:**
- Weight magnitude can be high due to initialization, not importance
- Gradient norm indicates "the loss cares about this block"
- EMA smoothing filters out batch-specific noise

### 5.4 Score Step (Level 1, Every ~10 Steps)

```python
def score_step(self):
    # Normalize accumulators by step count
    if self.acc_steps > 0:
        self.activation_norm_acc /= self.acc_steps
        self.error_norm_acc /= self.acc_steps
    
    # Age all blocks (they've survived another scoring period)
    self.block_age += 1
    
    # Reset step counter (but NOT the accumulators—keep for Level 2)
    self.acc_steps.zero_()
```

### 5.5 Topology Step (Level 2, Every ~100 Steps)

This is the core algorithm. Two variants:

#### 5.5.1 Magnitude-Based (Fallback/Baseline)

```python
def topology_step_magnitude(self):
    # Candidate scores: outer product of error × activation
    # High error at output r AND high activation at input c
    # suggests connecting r to c would help
    candidate_scores = outer(self.error_norm_acc, self.activation_norm_acc)  # [R, C]
    
    num_swaps = 0
    
    for r in range(R):
        # Current block scores (accumulated gradient importance)
        current_scores = self.block_score_ema[r]  # [K]
        current_cols = self.col_indices[r].tolist()
        
        # Find weakest current block
        worst_k = argmin(current_scores)
        worst_score = current_scores[worst_k]
        worst_col = current_cols[worst_k]
        
        # Find strongest candidate not already active
        candidate_mask = ones(C, dtype=bool)
        for c in current_cols:
            candidate_mask[c] = False
        
        masked_scores = where(candidate_mask, candidate_scores[r], -inf)
        best_col = argmax(masked_scores)
        best_score = masked_scores[best_col]
        
        # Swap if candidate is significantly better (1.5× threshold)
        # The threshold prevents oscillation from noise
        if best_score > worst_score * 1.5:
            # Update topology
            self.col_indices[r, worst_k] = best_col
            
            # Initialize new block weights
            self.values.data[r, worst_k].zero_()
            kaiming_init_(self.values.data[r, worst_k:worst_k+1])
            self.values.data[r, worst_k] *= 0.1  # Conservative scale
            
            # Reset age for new block
            self.block_age[r, worst_k] = 0
            
            num_swaps += 1
    
    # Reset all accumulators for next cycle
    self.block_score_ema.zero_()
    self.activation_norm_acc.zero_()
    self.error_norm_acc.zero_()
    
    return num_swaps
```

#### 5.5.2 Learned Scoring (Target Implementation)

```python
def topology_step_learned(self, generator: torch.Generator):
    # === 1. Build features for existing blocks ===
    existing_features = []
    
    # Precompute structural features
    row_densities = (self.col_indices >= 0).float().mean(dim=1)  # [R]
    col_counts = bincount(self.col_indices.flatten(), minlength=C).float()
    col_popularity = col_counts / col_counts.sum()  # [C]
    
    for r in range(R):
        for k in range(K):
            c = self.col_indices[r, k].item()
            features = tensor([
                self.block_score_ema[r, k],           # Gradient importance (0, ~10)
                self.block_age[r, k] / 100.0,         # Normalized age (0, ~1+)
                row_densities[r],                      # How full is this row (0, 1)
                col_popularity[c],                     # How popular is this column (0, 1)
            ])
            existing_features.append(features)
    
    existing_features = stack(existing_features)  # [R*K, 4]
    
    # === 2. Sample and build features for candidates ===
    num_candidates = K  # Sample as many candidates as we have slots
    
    # Sample columns weighted by activation norm (focus exploration on active inputs)
    probs = self.activation_norm_acc / (self.activation_norm_acc.sum() + 1e-8)
    candidate_cols = multinomial(
        probs.expand(R, C), 
        num_candidates,
        replacement=True,
        generator=generator  # DDP determinism
    )  # [R, num_candidates]
    
    # Candidate proxy scores (error × activation)
    candidate_proxy = outer(self.error_norm_acc, self.activation_norm_acc)  # [R, C]
    
    candidate_features = []
    for r in range(R):
        for k in range(num_candidates):
            c = candidate_cols[r, k].item()
            features = tensor([
                candidate_proxy[r, c],                 # Proxy score
                0.0,                                   # Age = 0 (new)
                row_densities[r],                      # Row density
                col_popularity[c],                     # Column popularity
            ])
            candidate_features.append(features)
    
    candidate_features = stack(candidate_features)  # [R*num_candidates, 4]
    
    # === 3. Score all blocks with learned controller ===
    all_features = cat([existing_features, candidate_features], dim=0)
    all_scores = self.topology_controller(all_features)  # [R*K + R*num_candidates]
    
    existing_scores = all_scores[:R*K].view(R, K)
    candidate_scores = all_scores[R*K:].view(R, num_candidates)
    
    # === 4. Select top-K per row ===
    combined_scores = cat([existing_scores, candidate_scores], dim=1)  # [R, K + num_candidates]
    combined_cols = cat([self.col_indices, candidate_cols], dim=1)     # [R, K + num_candidates]
    
    _, top_indices = topk(combined_scores, K, dim=1)  # [R, K]
    new_col_indices = gather(combined_cols, 1, top_indices)
    
    # === 5. Reallocate weights ===
    new_values = zeros_like(self.values)
    new_ages = zeros_like(self.block_age)
    
    num_swaps = 0
    for r in range(R):
        for new_k in range(K):
            source_idx = top_indices[r, new_k].item()
            
            if source_idx < K:
                # Existing block survives: copy weights and age
                old_k = source_idx
                new_values[r, new_k] = self.values[r, old_k]
                new_ages[r, new_k] = self.block_age[r, old_k]
            else:
                # New block: initialize fresh
                kaiming_init_(new_values[r, new_k:new_k+1])
                new_values[r, new_k] *= 0.1  # Conservative
                new_ages[r, new_k] = 0
                num_swaps += 1
    
    # === 6. Commit changes ===
    self.values.data.copy_(new_values)
    self.col_indices.copy_(new_col_indices)
    self.block_age.copy_(new_ages)
    
    # === 7. Reset accumulators ===
    self.block_score_ema.zero_()
    self.activation_norm_acc.zero_()
    self.error_norm_acc.zero_()
    
    return num_swaps
```

---

## 6. Hardware Considerations

### 6.1 Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                 │
├─────────────────────────────────────────────────────────┤
│  Registers      │  256 KB/SM   │  ~1 cycle    │ Fastest │
│  Shared Memory  │  164 KB/SM   │  ~20 cycles  │         │
│  L1 Cache       │  128 KB/SM   │  ~30 cycles  │         │
│  L2 Cache       │  40 MB       │  ~200 cycles │         │
│  Global (HBM)   │  80 GB       │  ~400 cycles │ Slowest │
└─────────────────────────────────────────────────────────┘
```

**Block-ELL exploits this by:**

1. **Contiguous weight access**: `values[r, k, :, :]` is 256 contiguous floats (1 KB). Fits in L1, loads in one transaction.

2. **Predictable access pattern**: Iteration over k is regular. Hardware prefetcher can anticipate next block.

3. **Tile size matches cache**: 16×16 × 4 bytes = 1 KB. L1 cache line is 128 bytes, so 8 cache lines per tile.

### 6.2 Tensor Core Utilization

NVIDIA Tensor Cores (Volta+) accelerate small matrix multiplies:

```
WMMA instruction: C = A × B + C
  A: 16×16 matrix (FP16)
  B: 16×16 matrix (FP16)
  C: 16×16 matrix (FP32)
  
Throughput: 125 TFLOPS (A100) for these shapes
```

Our 16×16 tile size matches WMMA exactly. The block-sparse forward becomes a sequence of WMMA calls.

**However:** Our forward is [B, B] @ [B] (matrix-vector), not [B, B] @ [B, B] (matrix-matrix). Tensor Cores help less here. Larger batch sizes help amortize.

### 6.3 Memory Bandwidth Analysis

**Dense layer (640 → 2560):**
```
Weight load: 640 × 2560 × 2 bytes = 3.28 MB (FP16)
Input load:  batch × 640 × 2 bytes
Output store: batch × 2560 × 2 bytes

For batch=32: 3.28 MB weights dominate
Arithmetic intensity: (32 × 640 × 2560 × 2) / (3.28 MB) ≈ 31 FLOP/byte
```

**Block-sparse at 50% density:**
```
Weight load: 160 × 20 × 256 × 2 bytes = 1.64 MB
Index load:  160 × 20 × 4 bytes = 12.8 KB (negligible)
Input load:  batch × 640 × 2 bytes (same, need full input)
Output store: batch × 2560 × 2 bytes (same)

Arithmetic intensity: (32 × 160 × 20 × 256 × 2) / (1.64 MB) ≈ 31 FLOP/byte (same!)
```

**Key insight:** We're still bandwidth-bound, but we transfer half the data. Theoretical 2× speedup at 50% density.

**Reality:** Indirection through col_indices adds latency. Expect 1.5-1.8× actual speedup.

### 6.4 The Gather Problem

```python
input_blocks = x_blocked[:, col_indices[r], :]  # Indirect access
```

Each row has different column indices. This is a gather operation:
- Memory accesses are non-contiguous in input
- Cannot use vectorized loads across rows
- L2 cache helps if the same columns are reused across rows

**Mitigation strategies:**

1. **Column popularity sorting**: Place popular columns first in memory. More cache hits.

2. **Async prefetching (CUDA)**: Issue loads for k+1 while computing k.

3. **Batch along rows**: Process multiple rows with same column pattern together.

Triton handles basic optimization; CUDA would be needed for advanced pipelining.

---

## 7. Triton Implementation

### 7.1 Forward Kernel

```python
import triton
import triton.language as tl

@triton.jit
def block_ell_forward_kernel(
    # Pointers
    x_ptr,              # [batch, in_features]
    values_ptr,         # [R, K, B, B]
    col_indices_ptr,    # [R, K]
    output_ptr,         # [batch, out_features]
    # Dimensions
    batch_size,
    R: tl.constexpr,    # Number of output block-rows
    K: tl.constexpr,    # Active blocks per row
    C,                  # Number of input block-columns
    B: tl.constexpr,    # Block size
    # Strides (in elements, not bytes)
    stride_x_batch, stride_x_feat,
    stride_v_r, stride_v_k, stride_v_i, stride_v_j,
    stride_c_r, stride_c_k,
    stride_o_batch, stride_o_feat,
):
    """
    Block-sparse matmul forward pass.
    
    Grid: (batch_size, R)
    Each program computes output[batch, r*B : (r+1)*B]
    """
    # Program indices
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)
    
    # Output accumulator (one B-element vector per program)
    acc = tl.zeros((B,), dtype=tl.float32)
    
    # Offsets within a block [0, 1, 2, ..., B-1]
    offs_b = tl.arange(0, B)
    
    # Base pointers for this program
    col_base = col_indices_ptr + pid_row * stride_c_r
    val_base = values_ptr + pid_row * stride_v_r
    x_batch_base = x_ptr + pid_batch * stride_x_batch
    
    # Iterate over K active blocks
    for k in range(K):
        # Load column index
        col_idx = tl.load(col_base + k * stride_c_k)
        
        # Load input block: x[batch, col_idx*B : (col_idx+1)*B]
        x_block_ptr = x_batch_base + col_idx * B * stride_x_feat
        x_block = tl.load(x_block_ptr + offs_b * stride_x_feat)  # [B]
        
        # Load weight tile and compute matmul
        # values[row, k, :, :] is B×B
        # We compute: acc += values[row,k] @ x_block
        val_tile_base = val_base + k * stride_v_k
        
        for i in range(B):
            # Load row i of weight tile
            w_row_ptr = val_tile_base + i * stride_v_i
            w_row = tl.load(w_row_ptr + offs_b * stride_v_j)  # [B]
            
            # Dot product
            dot = tl.sum(w_row * x_block)
            
            # Accumulate into position i of output
            # Using masked store pattern
            acc = tl.where(offs_b == i, acc + dot, acc)
    
    # Store output block
    out_ptr = output_ptr + pid_batch * stride_o_batch + pid_row * B * stride_o_feat
    tl.store(out_ptr + offs_b * stride_o_feat, acc)


def block_ell_forward(
    x: torch.Tensor,          # [batch, in_features]
    values: torch.Tensor,     # [R, K, B, B]
    col_indices: torch.Tensor # [R, K]
) -> torch.Tensor:
    """Triton wrapper for block-sparse forward."""
    
    batch_size, in_features = x.shape
    R, K, B, _ = values.shape
    out_features = R * B
    C = in_features // B
    
    # Allocate output
    output = torch.zeros(batch_size, out_features, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = (batch_size, R)
    
    block_ell_forward_kernel[grid](
        x, values, col_indices, output,
        batch_size, R, K, C, B,
        x.stride(0), x.stride(1),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        col_indices.stride(0), col_indices.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output
```

### 7.2 Backward Kernel (grad_values)

```python
@triton.jit
def block_ell_backward_values_kernel(
    # Inputs
    grad_out_ptr,       # [batch, out_features]
    x_ptr,              # [batch, in_features]
    col_indices_ptr,    # [R, K]
    # Output
    grad_values_ptr,    # [R, K, B, B]
    # Dimensions
    batch_size,
    R: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    # Strides
    stride_go_batch, stride_go_feat,
    stride_x_batch, stride_x_feat,
    stride_c_r, stride_c_k,
    stride_gv_r, stride_gv_k, stride_gv_i, stride_gv_j,
):
    """
    Compute gradient w.r.t. values.
    
    grad_values[r,k,i,j] = Σ_batch grad_out[batch, r*B+i] * x[batch, col[r,k]*B+j]
    
    Grid: (R, K)
    Each program computes one B×B gradient tile.
    """
    pid_r = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Load column index for this block
    col_idx = tl.load(col_indices_ptr + pid_r * stride_c_r + pid_k * stride_c_k)
    
    # Output gradient tile accumulator
    offs_i = tl.arange(0, B)
    offs_j = tl.arange(0, B)
    
    # Initialize gradient tile to zero
    grad_tile = tl.zeros((B, B), dtype=tl.float32)
    
    # Accumulate over batch
    for b in range(batch_size):
        # Load grad_out block: [B]
        go_ptr = grad_out_ptr + b * stride_go_batch + pid_r * B * stride_go_feat
        grad_out_block = tl.load(go_ptr + offs_i * stride_go_feat)  # [B]
        
        # Load input block: [B]
        x_ptr_b = x_ptr + b * stride_x_batch + col_idx * B * stride_x_feat
        x_block = tl.load(x_ptr_b + offs_j * stride_x_feat)  # [B]
        
        # Outer product: grad_out_block[:, None] * x_block[None, :]
        # Accumulate into grad_tile
        grad_tile += grad_out_block[:, None] * x_block[None, :]
    
    # Store gradient tile
    gv_base = grad_values_ptr + pid_r * stride_gv_r + pid_k * stride_gv_k
    for i in range(B):
        for j in range(B):
            ptr = gv_base + i * stride_gv_i + j * stride_gv_j
            tl.store(ptr, grad_tile[i, j])
```

### 7.3 Backward Kernel (grad_input)

```python
@triton.jit  
def block_ell_backward_input_kernel(
    # Inputs
    grad_out_ptr,       # [batch, out_features]
    values_ptr,         # [R, K, B, B]
    col_indices_ptr,    # [R, K]
    # Output (atomic adds)
    grad_x_ptr,         # [batch, in_features]
    # Dimensions
    batch_size,
    R: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    # Strides
    stride_go_batch, stride_go_feat,
    stride_v_r, stride_v_k, stride_v_i, stride_v_j,
    stride_c_r, stride_c_k,
    stride_gx_batch, stride_gx_feat,
):
    """
    Compute gradient w.r.t. input.
    
    grad_x[batch, col[r,k]*B+j] += Σ_i values[r,k,i,j] * grad_out[batch, r*B+i]
    
    Grid: (batch_size, R)
    Uses atomic adds because multiple (r,k) pairs may write to same input column.
    """
    pid_batch = tl.program_id(0)
    pid_r = tl.program_id(1)
    
    offs_b = tl.arange(0, B)
    
    # Load grad_out block for this (batch, row)
    go_base = grad_out_ptr + pid_batch * stride_go_batch + pid_r * B * stride_go_feat
    grad_out_block = tl.load(go_base + offs_b * stride_go_feat)  # [B]
    
    # Process each active block in this row
    for k in range(K):
        # Load column index
        col_idx = tl.load(col_indices_ptr + pid_r * stride_c_r + k * stride_c_k)
        
        # Load weight tile: [B, B]
        val_base = values_ptr + pid_r * stride_v_r + k * stride_v_k
        
        # Compute: grad_x_contrib[j] = Σ_i values[i,j] * grad_out[i]
        # This is values^T @ grad_out
        grad_x_contrib = tl.zeros((B,), dtype=tl.float32)
        
        for j in range(B):
            # Load column j of weight tile
            col_sum = 0.0
            for i in range(B):
                w_ij = tl.load(val_base + i * stride_v_i + j * stride_v_j)
                col_sum += w_ij * grad_out_block[i]
            grad_x_contrib = tl.where(offs_b == j, grad_x_contrib + col_sum, grad_x_contrib)
        
        # Atomic add to grad_x (multiple rows may contribute to same column)
        gx_ptr = grad_x_ptr + pid_batch * stride_gx_batch + col_idx * B * stride_gx_feat
        tl.atomic_add(gx_ptr + offs_b * stride_gx_feat, grad_x_contrib)
```

### 7.4 Autograd Integration

```python
class BlockELLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, values, col_indices):
        output = block_ell_forward(x, values, col_indices)
        ctx.save_for_backward(x, values, col_indices)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, values, col_indices = ctx.saved_tensors
        
        batch_size, in_features = x.shape
        R, K, B, _ = values.shape
        
        # Gradient w.r.t. values
        grad_values = torch.zeros_like(values)
        grid_gv = (R, K)
        block_ell_backward_values_kernel[grid_gv](
            grad_output, x, col_indices, grad_values,
            batch_size, R, K, B,
            grad_output.stride(0), grad_output.stride(1),
            x.stride(0), x.stride(1),
            col_indices.stride(0), col_indices.stride(1),
            grad_values.stride(0), grad_values.stride(1), 
            grad_values.stride(2), grad_values.stride(3),
        )
        
        # Gradient w.r.t. input
        grad_x = torch.zeros_like(x)
        grid_gx = (batch_size, R)
        block_ell_backward_input_kernel[grid_gx](
            grad_output, values, col_indices, grad_x,
            batch_size, R, K, B,
            grad_output.stride(0), grad_output.stride(1),
            values.stride(0), values.stride(1), values.stride(2), values.stride(3),
            col_indices.stride(0), col_indices.stride(1),
            grad_x.stride(0), grad_x.stride(1),
        )
        
        # No gradient for col_indices (discrete)
        return grad_x, grad_values, None
```

---

## 8. CMS Integration

### 8.1 Integration with DeepNestedOptimizer

The optimizer needs to:
1. Discover all CMSBlockLinear layers
2. Call score accumulation after each backward
3. Call score_step at Level 1 frequency
4. Call topology_step at Level 2 frequency

```python
# In DeepNestedOptimizer.__init__:

def __init__(self, model, ...):
    # ... existing initialization ...
    
    # Discover block-sparse layers
    self.block_sparse_layers: List[CMSBlockLinear] = []
    for module in model.modules():
        if isinstance(module, CMSBlockLinear):
            self.block_sparse_layers.append(module)
    
    # Create shared topology controller if we have sparse layers
    if self.block_sparse_layers:
        self.topology_controller = TopologyController(hidden_dim=32).to(self.device)
        
        # Share controller across all layers
        for layer in self.block_sparse_layers:
            layer.topology_controller = self.topology_controller
        
        # Add controller to meta-optimizer
        self.meta_optimizer.add_param_group({
            'params': self.topology_controller.parameters(),
            'lr': self.meta_lr,
        })
    else:
        self.topology_controller = None
```

### 8.2 Step Integration

```python
# In DeepNestedOptimizer.step():

def step(self, loss_value: Optional[float] = None):
    # ... existing step logic ...
    
    # Base optimizer step
    self.base_optimizer.step()
    
    # === Block-sparse score accumulation (every step) ===
    for layer in self.block_sparse_layers:
        layer.accumulate_scores()
    
    # === Level 1: Score step (every 10 steps) ===
    if self.global_step % 10 == 0:
        for layer in self.block_sparse_layers:
            layer.score_step()
    
    # === Level 2: Topology step (every 100 steps) ===
    if self.global_step % 100 == 0:
        # Deterministic generator for DDP synchronization
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.global_step)  # Same seed on all ranks
        
        total_swaps = 0
        for layer in self.block_sparse_layers:
            swaps = layer.topology_step(generator=generator)
            total_swaps += swaps
        
        if total_swaps > 0:
            self._log_metric('topology/total_swaps', total_swaps)
    
    self.global_step += 1
    
    # ... rest of existing step logic ...
```

### 8.3 Error Gradient Tracking

The error_norm_acc requires tracking gradients at layer outputs. Two approaches:

**Approach A: Hook-based (cleaner)**
```python
# In CMSBlockLinear.__init__:
def __init__(self, ...):
    # ... existing init ...
    
    # Register backward hook to capture output gradients
    self._output_grad_hook = None

def _register_hooks(self):
    """Call after layer is added to model."""
    def backward_hook(module, grad_input, grad_output):
        if self.training and grad_output[0] is not None:
            with torch.no_grad():
                grad_blocked = grad_output[0].view(-1, self.R, self.B)
                self.error_norm_acc += grad_blocked.norm(dim=(0, 2))
    
    self._output_grad_hook = self.register_full_backward_hook(backward_hook)
```

**Approach B: Explicit call (simpler, used in forward impl above)**
```python
# Tracked during forward via activation norms
# Error norms approximated from values.grad (less accurate but simpler)
```

Recommend Approach A for accuracy, Approach B for initial prototype.

---

## 9. Learned Topology Controller

### 9.1 Architecture

```python
class TopologyController(nn.Module):
    """
    Learned scoring function for topology decisions.
    
    Input features (per block):
        - score_ema: Accumulated gradient importance [0, ~10]
        - age: Normalized block age [0, ~1]  
        - row_density: Fraction of row capacity used [0, 1]
        - col_popularity: How often this column is used globally [0, 1]
    
    Output:
        - score: Higher = more likely to keep/add
    
    Training:
        - Meta-learning via validation loss
        - Topology decisions at step T evaluated at step T+k
    """
    
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        
        input_dim = 4  # [score_ema, age, row_density, col_popularity]
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to approximately preserve score_ema ranking."""
        with torch.no_grad():
            # Zero out final layer weights
            final_layer = self.net[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
            
            # Set weight for score_ema feature (index 0) to pass through
            # This makes initial behavior ≈ magnitude-based scoring
            final_layer.weight.data[0, 0] = 1.0
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Score blocks for keep/add decisions.
        
        Args:
            features: [N, 4] tensor of block features
        
        Returns:
            scores: [N] tensor of scores (higher = keep/add)
        """
        return self.net(features).squeeze(-1)
```

### 9.2 Meta-Learning Integration

The topology controller is trained alongside the momentum MLP and LR controller:

```python
# In DeepNestedOptimizer.meta_update():

def meta_update(self, val_batch, train_batches=None, loss_fn=None):
    """Update meta-learned components including topology controller."""
    
    # ... existing meta-update logic ...
    
    # The topology controller gradients flow through:
    # 1. topology_step() calls controller.forward()
    # 2. Selected blocks affect forward pass
    # 3. Forward pass on val_batch produces loss
    # 4. loss.backward() flows gradients to controller
    
    # Note: Topology decisions are discrete (top-k selection)
    # Gradients flow through the scores, not the selection
    # This is similar to attention mechanisms
```

**Gradient flow challenge:** The top-k selection is non-differentiable. Options:

1. **Straight-through estimator**: Pretend selection is identity in backward pass
2. **Soft selection**: Use softmax + weighted sum instead of hard top-k (training only)
3. **REINFORCE**: Treat selection as sampling, use policy gradient
4. **Score-only training**: Train controller to produce "good" scores, let selection be discrete

Recommend option 4 for simplicity. The controller learns that high scores should go to blocks that help validation loss. The actual selection mechanism (top-k) doesn't need to be differentiable.

### 9.3 Training Signal

```python
# Meta-objective for topology controller:
#
# At step T (topology decision time):
#   1. Controller scores existing + candidate blocks
#   2. Top-K selection determines new topology
#   3. Training continues for k steps with new topology
#   4. Validation loss at step T+k is the meta-objective
#
# The controller should learn:
#   - High scores for blocks that will contribute to low val loss
#   - Low scores for blocks that are dead weight
#   - Appropriate exploration (sometimes add uncertain candidates)

def compute_topology_meta_loss(self, val_batch, loss_fn):
    """Compute meta-loss for topology controller training."""
    
    # This is called during meta_update, after topology_step
    # The topology has already been updated based on controller scores
    
    # Forward pass on validation data
    val_loss = loss_fn(self.model, val_batch)
    
    # The gradient of val_loss w.r.t. controller parameters flows through:
    # val_loss -> model weights -> topology selection -> controller scores
    #
    # But topology selection (argmax/topk) is non-differentiable
    # So we use a proxy: train controller to minimize val_loss directly
    # by correlating score predictions with observed usefulness
    
    return val_loss
```

---

## 10. TitanMAC Integration

### 10.1 Configuration

```python
@dataclass
class TitanMACConfig:
    # ... existing config ...
    
    # Block sparsity settings
    use_block_sparse: bool = False
    block_sparse_tile_size: int = 16
    block_sparse_density: float = 0.5
    block_sparse_layers: Tuple[int, ...] = (8, 9, 10, 11, 12)  # Which layers
    block_sparse_components: Tuple[str, ...] = ('mlp',)  # Which components
```

### 10.2 MLP Modification

```python
# In titans_core/blocks/mlp.py

class TitanMLP(nn.Module):
    def __init__(self, config: TitanMACConfig, layer_idx: int):
        super().__init__()
        
        use_sparse = (
            config.use_block_sparse 
            and layer_idx in config.block_sparse_layers
            and 'mlp' in config.block_sparse_components
        )
        
        if use_sparse:
            self.fc1 = CMSBlockLinear(
                config.d_model,
                config.d_ff,
                tile_size=config.block_sparse_tile_size,
                density=config.block_sparse_density,
            )
            self.fc2 = CMSBlockLinear(
                config.d_ff,
                config.d_model,
                tile_size=config.block_sparse_tile_size,
                density=config.block_sparse_density,
            )
        else:
            self.fc1 = nn.Linear(config.d_model, config.d_ff)
            self.fc2 = nn.Linear(config.d_ff, config.d_model)
        
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
```

### 10.3 Initialization

```python
# In TitanMAC._init_weights():

def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, CMSBlockLinear):
        # Block-sparse uses Kaiming init internally
        # But we may want to match the 0.02 std for consistency
        # Scale: std = 0.02 corresponds to variance = 0.0004
        # Kaiming for ReLU: var = 2/fan_in
        # Our fan_in is reduced by density, so scale accordingly
        effective_fan_in = module.C * module.B * module.B * module.get_density()
        target_std = 0.02
        scale = target_std * math.sqrt(effective_fan_in / 2)
        module.values.data *= scale / module.values.data.std()
```

### 10.4 Training Script Modifications

```python
# In train_titanmac_nested.py

def create_optimizer(model, config):
    optimizer = DeepNestedOptimizer(
        model=model,
        base_lr=config.learning_rate,
        # ... existing args ...
    )
    
    # Log block-sparse layer info
    if optimizer.block_sparse_layers:
        logger.info(f"Block-sparse layers: {len(optimizer.block_sparse_layers)}")
        for i, layer in enumerate(optimizer.block_sparse_layers):
            logger.info(f"  Layer {i}: {layer.R}×{layer.K}×{layer.B}² "
                       f"({layer.get_density()*100:.1f}% dense)")
    
    return optimizer

def training_step(batch, model, optimizer, step):
    # Forward
    logits, aux_loss = model(batch['input_ids'], return_aux_loss=True)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                           batch['labels'].view(-1))
    loss = loss + aux_loss
    
    # Backward
    loss.backward()
    
    # Optimizer step (includes CMS level checks)
    result = optimizer.step(loss_value=loss.item())
    optimizer.zero_grad()
    
    # Log topology metrics periodically
    if step % 100 == 0 and optimizer.block_sparse_layers:
        for i, layer in enumerate(optimizer.block_sparse_layers):
            stats = layer.get_topology_stats()
            logger.info(f"Layer {i} topology: density={stats['density']:.3f}, "
                       f"avg_score={stats['avg_block_score']:.4f}, "
                       f"avg_age={stats['avg_block_age']:.1f}")
    
    return loss.item(), result
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

```python
# tests/test_block_sparse.py

class TestBlockELLForward:
    """Test forward pass correctness."""
    
    def test_dense_equivalence(self):
        """Block-sparse with 100% density should match dense."""
        torch.manual_seed(42)
        
        in_features, out_features = 64, 128
        batch_size = 8
        tile_size = 16
        
        # Create dense layer
        dense = nn.Linear(in_features, out_features, bias=False)
        
        # Create sparse layer with 100% density
        sparse = CMSBlockLinear(in_features, out_features, 
                                tile_size=tile_size, density=1.0, bias=False)
        
        # Copy weights (need to rearrange for block format)
        # ... weight copying logic ...
        
        # Test forward
        x = torch.randn(batch_size, in_features)
        y_dense = dense(x)
        y_sparse = sparse(x)
        
        assert torch.allclose(y_dense, y_sparse, atol=1e-5)
    
    def test_gradient_flow(self):
        """Gradients should flow correctly through sparse layer."""
        sparse = CMSBlockLinear(64, 128, tile_size=16, density=0.5)
        x = torch.randn(4, 64, requires_grad=True)
        y = sparse(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert sparse.values.grad is not None
        assert sparse.values.grad.shape == sparse.values.shape
    
    def test_topology_update(self):
        """Topology update should swap blocks."""
        sparse = CMSBlockLinear(64, 128, tile_size=16, density=0.5)
        
        # Record initial topology
        initial_topology = sparse.col_indices.clone()
        
        # Simulate some gradient accumulation
        sparse.values.grad = torch.randn_like(sparse.values)
        for _ in range(10):
            sparse.accumulate_scores()
        sparse.score_step()
        
        # Run topology update
        num_swaps = sparse.topology_step()
        
        # Should have some swaps (with random init, some blocks will be weak)
        # But not necessarily—depends on threshold
        final_topology = sparse.col_indices.clone()
        
        # Topology should be valid (no duplicates per row, valid indices)
        for r in range(sparse.R):
            cols = final_topology[r].tolist()
            assert len(set(cols)) == len(cols), "Duplicate columns in row"
            assert all(0 <= c < sparse.C for c in cols), "Invalid column index"


class TestTritonKernels:
    """Test Triton kernel correctness against PyTorch reference."""
    
    def test_forward_matches_reference(self):
        """Triton forward should match PyTorch reference implementation."""
        R, K, B, C = 8, 4, 16, 10
        batch_size = 4
        
        values = torch.randn(R, K, B, B, device='cuda')
        col_indices = torch.randint(0, C, (R, K), device='cuda', dtype=torch.int32)
        x = torch.randn(batch_size, C * B, device='cuda')
        
        # Triton implementation
        y_triton = block_ell_forward(x, values, col_indices)
        
        # PyTorch reference
        y_ref = block_ell_forward_reference(x, values, col_indices)
        
        assert torch.allclose(y_triton, y_ref, atol=1e-4)
    
    def test_backward_matches_autograd(self):
        """Triton backward should produce correct gradients."""
        R, K, B, C = 8, 4, 16, 10
        batch_size = 4
        
        values = torch.randn(R, K, B, B, device='cuda', requires_grad=True)
        col_indices = torch.randint(0, C, (R, K), device='cuda', dtype=torch.int32)
        x = torch.randn(batch_size, C * B, device='cuda', requires_grad=True)
        
        # Forward + backward with our implementation
        y = BlockELLFunction.apply(x, values, col_indices)
        loss = y.sum()
        loss.backward()
        
        grad_x_ours = x.grad.clone()
        grad_values_ours = values.grad.clone()
        
        # Reset grads
        x.grad = None
        values.grad = None
        
        # Forward + backward with reference (should use autograd)
        y_ref = block_ell_forward_reference(x, values, col_indices)
        loss_ref = y_ref.sum()
        loss_ref.backward()
        
        assert torch.allclose(grad_x_ours, x.grad, atol=1e-4)
        assert torch.allclose(grad_values_ours, values.grad, atol=1e-4)
```

### 11.2 Integration Tests

```python
class TestCMSIntegration:
    """Test integration with CMS optimizer."""
    
    def test_topology_updates_at_level2(self):
        """Topology should update every 100 steps."""
        model = create_small_model_with_sparse_layers()
        optimizer = DeepNestedOptimizer(model)
        
        topology_changes = []
        
        for step in range(250):
            # Simulate training step
            x = torch.randn(4, 64)
            y = model(x)
            loss = y.sum()
            loss.backward()
            
            result = optimizer.step(loss_value=loss.item())
            optimizer.zero_grad()
            
            if step % 100 == 99:  # After level 2 triggers
                for layer in optimizer.block_sparse_layers:
                    topology_changes.append(layer.col_indices.clone())
        
        # Should have 2 topology snapshots (step 99, 199)
        assert len(topology_changes) == 2
        # Topologies may or may not have changed (depends on dynamics)
    
    def test_ddp_determinism(self):
        """Topology decisions should be identical across DDP ranks."""
        # This requires multi-GPU setup, skip if not available
        if torch.cuda.device_count() < 2:
            pytest.skip("Need 2+ GPUs for DDP test")
        
        # ... DDP test implementation ...
```

### 11.3 Benchmark Tests

```python
class TestPerformance:
    """Benchmark sparse vs dense performance."""
    
    @pytest.mark.benchmark
    def test_forward_speedup(self):
        """Measure forward pass speedup at various densities."""
        in_features, out_features = 2560, 640
        batch_size = 32
        
        results = {}
        
        # Dense baseline
        dense = nn.Linear(in_features, out_features).cuda()
        x = torch.randn(batch_size, in_features).cuda()
        
        # Warmup
        for _ in range(10):
            _ = dense(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = dense(x)
        end.record()
        torch.cuda.synchronize()
        
        results['dense'] = start.elapsed_time(end) / 100
        
        # Sparse at various densities
        for density in [0.75, 0.5, 0.25, 0.1]:
            sparse = CMSBlockLinear(in_features, out_features, 
                                   density=density).cuda()
            
            # Warmup
            for _ in range(10):
                _ = sparse(x)
            torch.cuda.synchronize()
            
            # Benchmark
            start.record()
            for _ in range(100):
                _ = sparse(x)
            end.record()
            torch.cuda.synchronize()
            
            results[f'sparse_{density}'] = start.elapsed_time(end) / 100
        
        # Report
        for name, time_ms in results.items():
            speedup = results['dense'] / time_ms if 'sparse' in name else 1.0
            print(f"{name}: {time_ms:.3f} ms ({speedup:.2f}x)")
        
        # Assert we get SOME speedup at 50% density
        assert results['sparse_0.5'] < results['dense'] * 0.9
```

---

## 12. Success Metrics

### 12.1 Primary Metric: Catastrophic Forgetting Reduction

**Protocol:**
```
1. Train on Task A (e.g., mathematics) until convergence
   - Record Task A validation accuracy: A_before
   
2. Train on Task B (e.g., code) for N steps
   - Monitor Task B training loss
   
3. Re-evaluate Task A accuracy: A_after

4. Compute forgetting:
   Forgetting = (A_before - A_after) / A_before × 100%
```

**Targets:**
| Configuration | Expected Forgetting |
|---------------|---------------------|
| Dense baseline | 40-60% |
| Block-sparse (magnitude heuristic) | 25-40% |
| Block-sparse (learned controller) | 15-30% |

**Hypothesis validation:**
- Track which blocks are active for Task A vs Task B
- Expect: significant non-overlap (new pathways for new task)
- Expect: Task A pathways remain mostly intact

### 12.2 Secondary Metric: Wall-Clock Speedup

**Target:** 1.3-1.8× speedup at 50% density

**Measurement:**
```python
# Per-step time breakdown
forward_time:   X ms
backward_time:  Y ms  
optimizer_time: Z ms (including topology)

# Compare sparse vs dense
speedup = dense_total_time / sparse_total_time
```

**Important:** Only count if we get ACTUAL speedup. FLOP reduction without wall-clock improvement is a failure.

### 12.3 Tertiary Metrics

**Topology dynamics:**
- Blocks swapped per topology step (should be 1-10%, not 0% or 50%)
- Block age distribution (mix of old and new, not all ancient or all fresh)
- Column usage entropy (connections spread across inputs, not clustered)

**Training stability:**
- Loss curve smoothness (topology changes shouldn't cause spikes)
- Gradient norm stability (no explosions at topology boundaries)

**Memory efficiency:**
- Peak GPU memory vs dense baseline
- Should see reduction proportional to density

---

## Appendix A: File Structure

```
titans_core/
├── opt/
│   ├── deep_nested_optimizer.py   # Modified for block-sparse
│   ├── cms.py                     # Existing CMS (unchanged)
│   └── topology_controller.py     # NEW: Learned topology scorer
├── layers/
│   └── block_sparse.py            # NEW: CMSBlockLinear implementation
├── kernels/
│   ├── block_ell_forward.py       # NEW: Triton forward kernel
│   └── block_ell_backward.py      # NEW: Triton backward kernels
├── blocks/
│   └── mlp.py                     # Modified to use CMSBlockLinear
└── tests/
    ├── test_block_sparse.py       # NEW: Unit tests
    └── test_block_sparse_integration.py  # NEW: Integration tests
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Block-ELL** | Sparse matrix format with fixed number of blocks per row |
| **Tile** | A B×B block of weights (e.g., 16×16 = 256 parameters) |
| **Density** | Fraction of possible blocks that are active (K/C) |
| **Topology** | The pattern of active connections (which blocks exist) |
| **CMS** | Continuum Memory System - multi-timescale learning framework |
| **Level 0/1/2** | CMS frequency levels (every 1/10/100 steps) |
| **Score EMA** | Exponential moving average of block gradient norms |
| **WMMA** | Warp Matrix Multiply-Accumulate (Tensor Core instruction) |

---

## Appendix C: References

1. **RigL**: Evci et al., "Rigging the Lottery: Making All Tickets Winners" (2020)
2. **NEAT**: Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies" (2002)
3. **Lottery Tickets**: Frankle & Carlin, "The Lottery Ticket Hypothesis" (2019)
4. **Nested Learning**: Behrouz et al., "Nested Learning" (2024)
5. **Block Sparsity**: Gray et al., "GPU Kernels for Block-Sparse Weights" (2017)
6. **Learned Optimizers**: Andrychowicz et al., "Learning to Learn" (2016)

---

*End of Specification*
