# Data Model: CMS Dynamic Block Sparse Linear Layer

**Date**: 2025-12-25
**Branch**: `001-cms-block-sparse`

---

## Core Entities

### Block

A tile of weights that is the atomic unit of sparsity.

| Attribute | Type | Description |
|-----------|------|-------------|
| `row_idx` | int | Output block-row index [0, R) |
| `slot_idx` | int | Slot within row [0, K) |
| `col_idx` | int | Input block-column index [0, C) |
| `values` | Tensor[B, B] | Weight values (16×16 = 256 floats) |
| `score_ema` | float | Gradient importance EMA |
| `age` | int | Steps since creation/last swap |

**Relationships:**
- Block belongs to exactly one row (output block-row)
- Block connects to exactly one column (input block-column)
- Multiple blocks can connect to the same column (shared input)

**Validation Rules:**
- `0 <= row_idx < R`
- `0 <= slot_idx < K`
- `0 <= col_idx < C`
- `age >= 0`

---

### Topology

The pattern of active blocks in a layer.

| Attribute | Type | Description |
|-----------|------|-------------|
| `R` | int | Number of output block-rows |
| `C` | int | Number of input block-columns |
| `K` | int | Active blocks per row (fixed) |
| `B` | int | Block/tile size (default 16) |
| `col_indices` | Tensor[R, K] | Column index for each slot |
| `density` | float | K / C (active fraction) |

**Derived Properties:**
- `out_features = R × B`
- `in_features = C × B`
- `total_blocks = R × K`
- `total_parameters = R × K × B × B`

**Validation Rules:**
- `K <= C` (can't have more active blocks than columns)
- `density = K / C` (must match K exactly)
- All `col_indices[r]` values are unique within row r

**State Transitions:**
```
TopologyStep (every ~100 training steps):
  1. Score existing blocks
  2. Score candidate blocks
  3. Select top-K per row
  4. Swap low-scoring blocks for high-scoring candidates

Invariant: Exactly K blocks remain active per row (density unchanged)
```

---

### BlockScore

Per-block importance metric for topology decisions.

| Attribute | Type | Description |
|-----------|------|-------------|
| `gradient_ema` | Tensor[R, K] | EMA of gradient Frobenius norms |
| `activation_norm` | Tensor[C] | Accumulated input activation L2 norms |
| `error_norm` | Tensor[R] | Accumulated output gradient L2 norms |
| `block_age` | Tensor[R, K] | Steps since block creation |
| `acc_steps` | int | Steps since last Level 1 normalization |

**EMA Update Rule:**
```python
# After each backward pass:
instant_score = frobenius_norm(block.grad)
gradient_ema = α × instant_score + (1 - α) × gradient_ema
# α = 0.95 (20-step effective window)
```

**Candidate Scoring:**
```python
# For potential new blocks:
candidate_score[r, c] = error_norm[r] × activation_norm[c]
# High error at output r AND high activation at input c
# suggests connecting r to c would help learning
```

---

### TopologyDecision

A topology update event at CMS Level 2.

| Attribute | Type | Description |
|-----------|------|-------------|
| `step` | int | Training step when decision occurred |
| `num_swaps` | int | Number of blocks swapped |
| `before_topology` | Tensor[R, K] | col_indices before update |
| `after_topology` | Tensor[R, K] | col_indices after update |
| `pruned_blocks` | List[(r, k, c)] | Blocks that were removed |
| `grown_blocks` | List[(r, k, c)] | Blocks that were added |

**Logging Contract:**
- Log at every topology decision (every ~100 steps)
- Include swap rate: `num_swaps / (R × K) × 100%`
- Include entropy of column usage for diversity tracking

---

### CMSLevel

Multi-timescale update scheduling.

| Level | Frequency | Action |
|-------|-----------|--------|
| 0 | Every step | Weight updates (standard training) |
| 1 | Every ~10 steps | Score accumulation, age increment |
| 2 | Every ~100 steps | Topology decisions |

**State:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `global_step` | int | Total training steps |
| `level_counters` | Dict[int, int] | Steps since last trigger per level |
| `frequencies` | Dict[int, int] | Trigger frequency per level |

---

## Layer Configuration

### CMSBlockLinearConfig

Configuration for a block-sparse linear layer.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | int | required | Input dimension |
| `out_features` | int | required | Output dimension |
| `tile_size` | int | 16 | Block size (B) |
| `density` | float | 0.5 | Fraction of active blocks |
| `bias` | bool | True | Include bias term |
| `topology_freq` | int | 100 | Steps between topology decisions |
| `score_ema_alpha` | float | 0.95 | EMA momentum for gradient scores |
| `exploration_epsilon` | float | 0.05 | Random swap probability |
| `swap_threshold` | float | 1.5 | Required improvement ratio for swap |

**Constraints:**
- `in_features % tile_size == 0`
- `out_features % tile_size == 0`
- `0.1 <= density <= 1.0`
- `0.0 <= exploration_epsilon <= 0.5`

---

## Relationships Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       CMSBlockLinear                        │
├─────────────────────────────────────────────────────────────┤
│ config: CMSBlockLinearConfig                                │
│ topology: Topology                                          │
│ scores: BlockScore                                          │
│ values: Parameter[R, K, B, B]                               │
│ bias: Parameter[out_features] (optional)                    │
└─────────────────────────────────────────────────────────────┘
           │                              │
           │ has                          │ has
           ▼                              ▼
┌─────────────────────┐      ┌───────────────────────────────┐
│      Topology       │      │         BlockScore            │
├─────────────────────┤      ├───────────────────────────────┤
│ R, C, K, B          │      │ gradient_ema[R, K]            │
│ col_indices[R, K]   │      │ activation_norm[C]            │
│ density             │      │ error_norm[R]                 │
└─────────────────────┘      │ block_age[R, K]               │
           │                  └───────────────────────────────┘
           │ contains                     │
           ▼                              │ tracks
┌─────────────────────┐                   │
│       Block         │◄──────────────────┘
├─────────────────────┤
│ row_idx, slot_idx   │
│ col_idx             │
│ values[B, B]        │
│ score_ema, age      │
└─────────────────────┘
```

---

## State Machine: Block Lifecycle

```
┌──────────┐
│ INACTIVE │ ◄──────────────────────────────────────┐
│ (pruned) │                                        │
└────┬─────┘                                        │
     │ Grown (candidate selected)                   │
     │ → Initialize with Kaiming scaled ×0.1        │
     │ → age = 0                                    │
     │                                              │
     ▼                                              │
┌──────────┐                                        │
│  ACTIVE  │ ←─┐                                    │
│(training)│   │                                    │
└────┬─────┘   │                                    │
     │         │                                    │
     │ Topology step:                               │
     │ ├─ If in top-K by score → ACTIVE (continue)──┘
     │ │  → age += 1
     │ │
     │ └─ If not in top-K → INACTIVE (pruned) ──────┘
     │    → weights discarded
     │    → slot reused for new block
     │
     │ Training step:
     │ → weights updated by optimizer
     │ → gradient accumulated in score_ema
     └──────────────────────────────────────────────┐
                                                    │
                                                    ▼
                                            (loop continues)
```

---

## Tensor Shapes Reference

For TitanMAC MLP layers (d_model=640, d_ff=2560, tile_size=16, density=0.5):

### fc1 Layer (640 → 2560)

| Tensor | Shape | Size |
|--------|-------|------|
| `values` | [160, 20, 16, 16] | 0.82 MB |
| `col_indices` | [160, 20] | 12.8 KB |
| `gradient_ema` | [160, 20] | 12.8 KB |
| `activation_norm` | [40] | 160 B |
| `error_norm` | [160] | 640 B |
| `block_age` | [160, 20] | 12.8 KB |

### fc2 Layer (2560 → 640)

| Tensor | Shape | Size |
|--------|-------|------|
| `values` | [40, 80, 16, 16] | 0.82 MB |
| `col_indices` | [40, 80] | 12.8 KB |
| `gradient_ema` | [40, 80] | 12.8 KB |
| `activation_norm` | [160] | 640 B |
| `error_norm` | [40] | 160 B |
| `block_age` | [40, 80] | 12.8 KB |

Note: fc2 has more blocks per row (K=80) because input is larger (2560).
