# CMS Block-Sparse Quickstart

A guide to using CMSBlockLinear for dynamic block-sparse training with the Continuum Memory System (CMS).

## Overview

CMSBlockLinear is a drop-in replacement for `nn.Linear` that:
- Uses block-sparse weight storage (Block-ELL format)
- Dynamically adapts topology based on gradient importance
- Integrates with the CMS Level 2 update schedule

## Quick Start

### Basic Usage

```python
from titans_core.layers.block_sparse import CMSBlockLinear

# Create a sparse layer (in_features and out_features must be divisible by tile_size)
layer = CMSBlockLinear(
    in_features=128,   # Must be divisible by tile_size
    out_features=256,  # Must be divisible by tile_size
    tile_size=16,      # Block size (8, 16, 32, or 64 for Triton)
    density=0.5,       # Fraction of blocks to keep per row
    bias=True,
)

# Forward pass (same as nn.Linear)
x = torch.randn(32, 128)
y = layer(x)  # [32, 256]
```

### Converting from Dense

```python
import torch.nn as nn
from titans_core.layers.block_sparse import CMSBlockLinear

# Create or load an existing dense layer
dense = nn.Linear(128, 256)

# Convert to sparse using magnitude-based block selection
sparse = CMSBlockLinear.from_dense(
    dense,
    tile_size=16,
    density=0.5,  # Keep top 50% of blocks by magnitude
)
```

### Training Loop with Topology Updates

```python
model = MyModelWithBlockSparse()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step, batch in enumerate(dataloader):
    # Forward pass
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # Accumulate gradient scores (every step)
    for layer in model.get_block_sparse_layers():
        layer.accumulate_scores()

    optimizer.step()

    # Score step (Level 1) - every 10 steps
    if step % 10 == 0:
        for layer in model.get_block_sparse_layers():
            layer.score_step()

    # Topology step (Level 2) - every 100 steps
    if step % 100 == 0:
        for layer in model.get_block_sparse_layers():
            result = layer.topology_step(global_step=step)
            print(f"Swapped {result.num_swaps} blocks")
```

## Configuration

### TitanMACConfig with Block-Sparse

```python
from titans_core.config import TitanMACConfig

config = TitanMACConfig(
    vocab_size=50257,
    d_model=256,           # Must be divisible by tile_size
    n_heads=4,
    n_layers=6,
    d_ff=1024,             # Must be divisible by tile_size
    max_seq_len=512,

    # Block-sparse settings (if integrated)
    use_block_sparse=True,
    block_sparse_density=0.5,
    block_sparse_tile_size=16,
)
```

### DeepNestedOptimizer Integration

```python
from titans_core.opt import DeepNestedOptimizer

optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=1e-3,
    meta_lr=1e-4,
    cms_frequencies=[1, 10, 100],  # L0=every step, L1=10, L2=100
    mode='simple',
)

# The optimizer can manage block-sparse layers if registered
```

## CMS Update Schedule

The Continuum Memory System uses a multi-frequency update schedule:

| Level | Frequency | Operation | Purpose |
|-------|-----------|-----------|---------|
| L0 | Every step | `accumulate_scores()` | Collect gradient norms |
| L1 | Every 10 steps | `score_step()` | Normalize, increment ages |
| L2 | Every 100 steps | `topology_step()` | Prune/grow blocks |

### Customizing the Schedule

```python
# Faster adaptation (more topology changes)
score_interval = 5
topology_interval = 50

# Slower adaptation (more stable)
score_interval = 20
topology_interval = 200
```

## Monitoring

### Topology Statistics

```python
stats = layer.get_topology_stats()
print(f"Density: {stats.density:.2%}")
print(f"Column entropy: {stats.column_entropy:.3f}")
print(f"Avg block score: {stats.avg_block_score:.4f}")
print(f"Avg block age: {stats.avg_block_age:.1f}")
print(f"Total blocks: {stats.num_blocks}")
```

### Swap Rate History

```python
avg_swap_rate = layer.get_avg_swap_rate()
print(f"Average swap rate: {avg_swap_rate:.2%}")
```

### Block Age Distribution

```python
age_dist = layer.get_block_age_distribution()
# {0: 10, 1: 15, 2: 8, ...}  # age -> count
```

## Troubleshooting

### Dimension Errors

**Error:** `in_features (65) must be divisible by tile_size (16)`

**Solution:** Ensure dimensions are multiples of tile_size:
```python
# Round up to nearest multiple
tile_size = 16
d_model = ((d_model + tile_size - 1) // tile_size) * tile_size
```

### No Speedup

**Symptoms:** Sparse layer not faster than dense

**Causes and solutions:**
1. **CPU execution**: Triton kernels require CUDA
   ```python
   model = model.cuda()
   ```

2. **Low density**: Speedup requires density < 0.7
   ```python
   layer = CMSBlockLinear(..., density=0.5)
   ```

3. **Small dimensions**: Block overhead dominates for small matrices
   - Use d_model >= 256 for measurable speedup

4. **Triton not installed**:
   ```bash
   pip install triton
   ```

### Training Instability

**Symptoms:** Loss spikes > 2x during training

**Solutions:**

1. **Reduce swap rate**: Increase topology_interval
   ```python
   topology_interval = 200  # Instead of 100
   ```

2. **Increase swap threshold**: Require larger improvement for swaps
   ```python
   # In TopologyScorer initialization
   swap_threshold = 2.0  # Default is 1.5
   ```

3. **Reduce exploration**: Lower epsilon-greedy probability
   ```python
   exploration_epsilon = 0.02  # Default is 0.05
   ```

4. **Smaller new block initialization**:
   The default is Kaiming x 0.1, which is usually safe.

### Memory Issues

**Symptoms:** OOM errors during topology_step

**Solutions:**

1. **Reduce topology history size**:
   ```python
   layer._topology_history_max_size = 5  # Default 10
   ```

2. **Disable snapshots**:
   ```python
   layer.topology_step(save_snapshot=False)
   ```

### DDP Topology Divergence

**Symptoms:** Different ranks have different topologies

**Solutions:**

1. **Use global_step for deterministic RNG**:
   ```python
   layer.topology_step(global_step=step)
   ```

2. **Verify checksum**:
   ```python
   checksum = layer.get_topology_checksum()
   # All ranks should have same checksum
   ```

3. **Manual sync** (if needed):
   ```python
   layer.sync_topology_scores()  # Called automatically in topology_step
   ```

## Performance Tips

1. **Use compatible tile sizes**: 16 is optimal for most GPUs (WMMA)
2. **Batch size**: Larger batches amortize kernel launch overhead
3. **torch.compile**: Works with block-sparse (but don't use fullgraph=True)
4. **Mixed precision**: Block-sparse supports AMP

## Example: Complete Training Script

See `examples/train_block_sparse.py` for a complete working example.

## API Reference

### CMSBlockLinear

```python
class CMSBlockLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_size: int = 16,
        density: float = 0.5,
        bias: bool = True,
        score_ema_alpha: float = 0.95,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...
    def accumulate_scores(self) -> None: ...
    def score_step(self) -> None: ...
    def topology_step(
        self,
        generator: Optional[torch.Generator] = None,
        save_snapshot: bool = True,
        global_step: Optional[int] = None,
    ) -> TopologyDecisionResult: ...
    def get_topology_stats(self) -> TopologyStats: ...
    def get_density(self) -> float: ...
    def to_dense(self) -> Tensor: ...

    @classmethod
    def from_dense(
        cls,
        dense_layer: nn.Linear,
        tile_size: int = 16,
        density: float = 0.5,
    ) -> "CMSBlockLinear": ...
```

### TopologyStats

```python
@dataclass
class TopologyStats:
    density: float           # K/C
    avg_block_score: float   # Mean gradient EMA
    avg_block_age: float     # Mean age in topology steps
    column_entropy: float    # 0-1 normalized
    num_blocks: int          # R * K
```

### TopologyDecisionResult

```python
@dataclass
class TopologyDecisionResult:
    num_swaps: int                        # Blocks swapped
    swap_rate: float                      # num_swaps / total_blocks
    pruned_positions: List[Tuple[int, int]]  # (row, slot) pruned
    grown_columns: List[int]              # New column indices
```
