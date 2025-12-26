# Quickstart: CMS Dynamic Block Sparse Linear Layer

**Date**: 2025-12-25
**Branch**: `001-cms-block-sparse`

---

## Installation

The block-sparse layer is part of `titans_core`. No additional installation needed.

```bash
pip install -e .
```

---

## Basic Usage

### 1. Create a Block-Sparse Layer

```python
from titans_core.layers import CMSBlockLinear

# Create a 640 → 2560 sparse layer at 50% density
layer = CMSBlockLinear(
    in_features=640,
    out_features=2560,
    tile_size=16,      # 16×16 blocks (WMMA compatible)
    density=0.5,       # 50% of blocks active per row
    bias=True,
)

# Forward pass works like nn.Linear
x = torch.randn(32, 640)  # [batch, in_features]
y = layer(x)              # [batch, out_features] = [32, 2560]
```

### 2. Training with Topology Updates

```python
# Training loop with manual topology management
for step, batch in enumerate(dataloader):
    # Forward + backward
    loss = model(batch)
    loss.backward()

    # Accumulate scores (every step)
    for layer in model.block_sparse_layers:
        layer.accumulate_scores()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Score step (every 10 steps)
    if step % 10 == 0:
        for layer in model.block_sparse_layers:
            layer.score_step()

    # Topology step (every 100 steps)
    if step % 100 == 0:
        for layer in model.block_sparse_layers:
            result = layer.topology_step()
            print(f"Step {step}: swapped {result.num_swaps} blocks")
```

---

## Integration with TitanMAC

### Configure Model for Block Sparsity

```python
from titans_core.config import TitanMACConfig
from titans_core.models import create_titanmac_model

# Configure which layers use block sparsity
config = TitanMACConfig(
    d_model=640,
    d_ff=2560,
    n_layers=16,

    # Block sparsity settings
    use_block_sparse=True,
    block_sparse_tile_size=16,
    block_sparse_density=0.5,
    block_sparse_layers=(8, 9, 10, 11, 12),  # Middle layers only
    block_sparse_components=('mlp',),         # Only MLP, not attention
)

# Create model (sparse layers automatically used)
model = create_titanmac_model(config)
```

### Train with DeepNestedOptimizer

The optimizer handles topology steps automatically:

```python
from titans_core.opt import DeepNestedOptimizer

optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    meta_lr=1e-4,
    # Topology updates happen at CMS Level 2 (every 100 steps)
)

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch)['loss']
    loss.backward()

    # optimizer.step() handles:
    # - Weight updates (Level 0)
    # - Score accumulation (every step)
    # - Score normalization (Level 1, every 10 steps)
    # - Topology decisions (Level 2, every 100 steps)
    optimizer.step(loss_value=loss.item())
```

---

## Monitoring Topology

### Log Topology Statistics

```python
# Get current topology state
stats = layer.get_topology_stats()
print(f"Density: {stats.density:.2%}")
print(f"Avg block score: {stats.avg_block_score:.4f}")
print(f"Avg block age: {stats.avg_block_age:.1f} steps")
print(f"Column entropy: {stats.column_entropy:.2%}")

# Log after topology step
result = layer.topology_step()
print(f"Swapped {result.num_swaps} blocks ({result.swap_rate:.1%})")
```

### Visualize Topology

```python
# Get topology as dense matrix for visualization
col_indices = layer.col_indices  # [R, K]

import matplotlib.pyplot as plt
import numpy as np

# Create binary mask showing active blocks
mask = np.zeros((layer.R, layer.C))
for r in range(layer.R):
    for k in range(layer.K):
        c = col_indices[r, k].item()
        mask[r, c] = 1

plt.imshow(mask, aspect='auto', cmap='Blues')
plt.xlabel('Input Blocks')
plt.ylabel('Output Blocks')
plt.title(f'Topology (density={layer.density:.0%})')
plt.colorbar(label='Active')
plt.show()
```

---

## Configuration Reference

### CMSBlockLinear Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | int | required | Input dimension |
| `out_features` | int | required | Output dimension |
| `tile_size` | int | 16 | Block size (must divide features) |
| `density` | float | 0.5 | Active blocks per row (0.1-1.0) |
| `bias` | bool | True | Include bias term |

### TitanMACConfig Block Sparse Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_block_sparse` | bool | False | Enable block sparsity |
| `block_sparse_tile_size` | int | 16 | Block size |
| `block_sparse_density` | float | 0.5 | Target density |
| `block_sparse_layers` | tuple | () | Which layer indices |
| `block_sparse_components` | tuple | ('mlp',) | Which components |

### Topology Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `topology_freq` | 100 | 50-500 | Steps between decisions |
| `score_ema_alpha` | 0.95 | 0.9-0.99 | Gradient EMA momentum |
| `exploration_epsilon` | 0.05 | 0.01-0.15 | Random swap probability |
| `swap_threshold` | 1.5 | 1.2-2.0 | Required improvement ratio |

---

## Performance Tips

### Achieving Speedup

1. **Use 50% density or lower** for meaningful speedup
2. **Batch size matters**: Larger batches amortize kernel launch overhead
3. **Profile first**: Use `torch.profiler` to identify bottlenecks

### Memory Savings

At 50% density on fc1 (640→2560):
- Dense: 6.55 MB
- Sparse: 0.82 MB + 13 KB indices = **0.83 MB (87% reduction)**

### DDP Training

Topology synchronization is automatic with DeepNestedOptimizer:

```python
# All ranks use same RNG seed for topology decisions
generator = torch.Generator()
generator.manual_seed(global_step)  # Same on all ranks

# Scores are all-reduced before decisions
dist.all_reduce(block_score_ema, op=dist.ReduceOp.AVG)
```

---

## Troubleshooting

### "Dimensions not divisible by tile_size"

```python
# Error: in_features=650 not divisible by tile_size=16
layer = CMSBlockLinear(650, 2560)  # ValueError!

# Solution: Pad or adjust dimensions
layer = CMSBlockLinear(640, 2560)  # 640 % 16 == 0 ✓
```

### "Training unstable after topology step"

Reduce exploration or increase swap threshold:

```python
# More conservative topology updates
layer.exploration_epsilon = 0.01  # Less random swaps
layer.swap_threshold = 2.0        # Require 2x improvement
```

### "No speedup observed"

Check that Triton kernels are being used:

```python
# Verify CUDA is available
assert torch.cuda.is_available()

# Check kernel selection
import titans_core.kernels.block_ell_forward as kernel
print(f"Using kernel: {kernel.__name__}")
```

---

## Next Steps

- See `specs/001-cms-block-sparse/spec.md` for full requirements
- See `specs/001-cms-block-sparse/research.md` for design decisions
- Run baseline experiments before comparing sparse performance
