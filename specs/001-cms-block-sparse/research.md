# Research: CMS Dynamic Block Sparse Linear Layer

**Date**: 2025-12-25
**Branch**: `001-cms-block-sparse`
**Purpose**: Resolve technical unknowns before implementation

---

## 1. Block-ELL Format Best Practices

### Decision: Row-major Block-ELL with Fixed-K Slots

**Storage Structure:**
```python
values:      [R, K, B, B]   # R=output_blocks, K=active_per_row, B=tile_size
col_indices: [R, K]         # int32, range [0, C), no sentinel values needed
```

**Rationale:**
- Row-major layout enables contiguous tile access (values[r, k, :, :] is 1KB)
- Fixed-K means every slot is filled—no sentinel values (-1) needed
- 16×16 tile = 1KB = 8 cache lines, perfect L1 cache fit

**Alternatives Considered:**
| Format | Index Overhead | GPU Fit | Rejected Because |
|--------|---------------|---------|------------------|
| CSR | High (1 idx/element) | Fair | 16x more index storage for fixed-K |
| COO | High (2 idx/element) | Fair | Even worse overhead |
| Sliced ELLPACK | Medium | Good | Unnecessary complexity for fixed-K |
| Block-ELL | Low (1 idx/block) | Best | **SELECTED** |

**Key Parameters:**
- Index type: `int32` (sufficient for 2^31 blocks, standard practice)
- Alignment: PyTorch default (128-byte boundaries)
- Memory savings at 50%: ~87.5% for weights (6.55MB → 0.82MB per fc1 layer)

---

## 2. Triton Block-Sparse Kernels

### Decision: PyTorch Reference First, Then Triton Optimization

**Forward Pass Pattern:**
```python
# Grid: (batch_size, R) - one output block-row per thread block
for k in range(K):
    col_idx = col_indices[r, k]              # Scalar load
    x_block = input[batch, col_idx*B:...]    # Gather (indirect)
    w_tile = values[r, k, :, :]              # Contiguous load
    output[batch, r*B:...] += w_tile @ x_block
```

**Backward Pass Patterns:**

1. **grad_values** (Grid: R×K):
   - Outer product accumulation: `grad_tile += grad_out ⊗ x_block`
   - No atomics needed—each (r,k) writes to unique location

2. **grad_input** (Grid: batch×R):
   - Scatter-add via `tl.atomic_add()`
   - Bottleneck: multiple (r,k) pairs may write to same input column
   - Mitigation: Start simple, measure contention, consider SplitK if >30% overhead

**Tile Size Decision:**
| Size | Performance | Memory | Decision |
|------|-------------|--------|----------|
| 8×8 | Lower | Less | Too fine-grained |
| 16×16 | Baseline | 1KB | **SELECTED** - WMMA compatible |
| 32×32 | ~1.3x faster | 4KB | Future optimization |
| 64×64 | ~1.5x faster | 16KB | Risk of L1 spillage |

**Rationale:**
- 16×16 matches WMMA instruction size (Tensor Core)
- TitanMAC dimensions (640, 2560) divide evenly by 16
- Conservative choice; can benchmark 32×32 later

**Performance Targets (from literature):**
- Forward: 1.3-1.8x speedup at 50% density
- Backward: May be slower due to atomic contention
- End-to-end: 1.2x speedup target is conservative/realistic

---

## 3. Dynamic Topology Scoring

### Decision: Magnitude-Based Scoring with EMA + Epsilon-Greedy Exploration

**Scoring Strategy:**

| Component | Method | Hyperparameter |
|-----------|--------|----------------|
| **Block Retention** | Gradient magnitude EMA | α = 0.95 (20-step window) |
| **Candidate Growth** | Activation × Error product | EMA with α = 0.99 |
| **Exploration** | Epsilon-greedy | ε = 0.05 (5% random) |
| **Swap Threshold** | Candidate must be 1.5x better | Prevents oscillation |

**EMA Update Formula:**
```python
block_score_ema = α * |grad[block]| + (1 - α) * block_score_ema
# With α = 0.95, effective window ≈ 20 steps
```

**Epsilon-Greedy Implementation:**
```python
if random() < epsilon:  # ε = 0.05
    # Random swap regardless of scores
    candidate = random_unused_column()
else:
    # Score-based swap
    candidate = argmax(candidate_scores)
```

**Alternatives Considered:**
| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| Pure magnitude | Simple | Miss gradient signal | Rejected |
| Pure gradient | Captures importance | Noisy | Combined with EMA |
| Wanda (|W| × ||X||) | State-of-art for pruning | Needs calibration data | Future work |
| Learned controller | Optimal in theory | Complex, unstable early | V2 (after heuristics work) |

**Key Hyperparameter Ranges:**
- Update period: 100 steps (Level 2 frequency)
- Prune fraction: 10% of blocks per decision
- Growth fraction: 10% (maintain density)
- EMA α for retention: 0.90-0.99 (use 0.95)
- EMA α for growth: 0.95-0.99 (use 0.99)
- Epsilon: 0.01-0.15 (use 0.05)

---

## 4. DDP Topology Synchronization

### Decision: Sync Scores at Level 2, Not Every Step

**Synchronization Strategy:**
```python
# At topology_step (every ~100 steps), BEFORE selection:
if dist.is_initialized():
    dist.all_reduce(block_score_ema, op=dist.ReduceOp.AVG)
    dist.all_reduce(activation_norm_acc, op=dist.ReduceOp.AVG)
    dist.all_reduce(error_norm_acc, op=dist.ReduceOp.AVG)

# Then use deterministic RNG:
generator = torch.Generator(device=self.device)
generator.manual_seed(global_step)  # Same seed on all ranks
```

**Rationale:**
- Score accumulation is local—different ranks see different micro-batches
- Only the final decision needs to be synchronized
- Syncing only at Level 2 adds one small all-reduce per 100 steps
- Overhead: 3200 floats × 4 bytes = 12.8KB per sync (negligible)

**Alternatives Considered:**
| Strategy | Overhead | Correctness | Decision |
|----------|----------|-------------|----------|
| Sync every step | High | Perfect | Overkill |
| Sync at Level 2 | Low | Sufficient | **SELECTED** |
| Broadcast from rank 0 | Low | Biased | Rejected |
| No sync (diverge) | None | Wrong | Invalid |

**Verification:**
- Add checksum comparison in tests: `assert all topologies match across ranks`
- Log topology divergence metric in training

---

## 5. Catastrophic Forgetting Baselines

### Decision: Establish Dense Baselines Before Sparse Experiments

**Required Baseline Experiments:**

| Experiment | Purpose | Steps | Output |
|------------|---------|-------|--------|
| **Dense Task A only** | Ground truth accuracy | 10K | A_baseline |
| **Dense Task A → B** | Forgetting rate | 10K + 10K | Forgetting % |
| **Dense with EWC** | SOTA comparison | 10K + 10K | EWC forgetting % |
| **Static sparse 50%** | Lower bound | 10K + 10K | Static forgetting % |

**Forgetting Measurement Protocol:**
```
1. Train on Task A until convergence → Record accuracy_A_before
2. Train on Task B for same steps → Record accuracy_B_final
3. Re-evaluate Task A → Record accuracy_A_after
4. Forgetting = (accuracy_A_before - accuracy_A_after) / accuracy_A_before × 100%
```

**Expected Baselines (from literature):**
- Dense model: 40-60% forgetting
- EWC: 25-40% forgetting
- Static sparse: ~same as dense or worse
- Dynamic block-sparse target: <30% forgetting (hypothesis to validate)

**Pathway Overlap Measurement:**
```python
# For each block, record activation during Task A vs Task B
# Overlap = |blocks_active_A ∩ blocks_active_B| / |blocks_active_A ∪ blocks_active_B|
# Target: overlap < 70% (distinct pathways for distinct tasks)
```

**Tasks for Testing:**
- Task A: Mathematics (existing `train_math.py`)
- Task B: NLP/Language modeling (existing `train_titanmac_nested.py` with FineWeb)
- Both already have training infrastructure in the codebase

---

## Summary of Decisions

| Topic | Decision | Key Rationale |
|-------|----------|---------------|
| **Format** | Block-ELL, fixed-K, int32 indices | 16x lower index overhead than CSR |
| **Tile Size** | 16×16 | WMMA compatible, conservative start |
| **Kernels** | PyTorch reference first, Triton later | Correctness before optimization |
| **Scoring** | Gradient EMA + epsilon-greedy | Balance exploitation/exploration |
| **EMA α** | 0.95 (retention), 0.99 (growth) | 20-step and 100-step windows |
| **Epsilon** | 0.05 | 5% random exploration |
| **DDP Sync** | At Level 2 only | Minimal overhead, sufficient correctness |
| **Baselines** | Dense + EWC + Static before sparse | Scientific rigor requires comparison |

---

## Open Questions for Implementation

1. **Atomic contention in grad_input**: Profile to determine if SplitK is needed
2. **32×32 tiles on H100**: Benchmark after 16×16 works correctly
3. **Epsilon decay**: Start fixed, consider decay schedule if exploration hurts late training
4. **Block age factor**: Should old blocks be protected from pruning? (Start without, add if instability)

---

## Sources

- [RigL: Rigging the Lottery - Making All Tickets Winners (Evci et al., 2020)](https://arxiv.org/abs/1911.11134)
- [Wanda: A Simple and Effective Pruning Approach for LLMs (Sun et al., 2024)](https://arxiv.org/abs/2306.11695)
- [MegaBlocks: Efficient Sparse Training (Gale et al., 2023)](https://arxiv.org/abs/2211.15841)
- [NVIDIA Block Sparse Tensor Cores](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)
- [Triton Block-Scaled MatMul Tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [cuSPARSE Storage Formats](https://docs.nvidia.com/cuda/cusparse/storage-formats.html)
