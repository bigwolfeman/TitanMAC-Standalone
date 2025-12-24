# Remaining Performance Gains for TitanMAC + DeepNestedOptimizer

**Date**: December 20, 2025
**Current Performance**: 3.5-3.8 it/s (after ~200 step torch.compile warmup)
**Hardware**: NVIDIA RTX 5090, PyTorch 2.9.1+cu128

This document catalogs optimization opportunities identified from profiling and analysis of the modded-nanogpt repository.

---

## Already Implemented

| Optimization | Location | Status |
|--------------|----------|--------|
| torch.compile (dynamic=False) | train_titanmac_nested.py | Done (fullgraph incompatible with neural memory) |
| compiled_autograd | train_titanmac_nested.py | Done (better backward optimization) |
| AMP (mixed precision) | train_titanmac_nested.py | Done |
| TensorFloat32 matmul | train_titanmac_nested.py | Done |
| Fused gradient clipping | deep_nested_optimizer.py | Done + compiled |
| Fused _compute_group_stats | deep_nested_optimizer.py | Done |
| Controller batching (every 10 steps) | train_titanmac_nested.py | Done |
| NaN guards (stats, controller, gradients) | deep_nested_optimizer.py, nested_controller.py | Done |
| donated_buffer=False for neural memory | train_titanmac_nested.py | Done |
| Compiled preprocess_gradient | deep_nested_optimizer.py | Done |

---

## Tier 1: High Impact, Low Effort

### 1. Reduce Evaluation Overhead

**Problem**: Every 50 steps, training blocks for 100 eval batches.

**Current Impact**: Evaluation takes ~66% of wall-clock time!

**Solution**: Use CLI args to reduce eval frequency:
```bash
python train_titanmac_nested.py --eval_every 100 --eval_steps 30 --speedy
```

**Expected Gain**: 20-30% wall-clock speedup

---

### 2. Remove Redundant .contiguous() Calls

**Location**: train_titanmac_nested.py:443-444

```python
# Current (unnecessary copies)
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = y[:, 1:].contiguous()

# Better (PyTorch handles this internally)
shift_logits = logits[:, :-1, :]
shift_labels = y[:, 1:]
```

**Expected Gain**: 2-4% (avoids ~400MB copy per step)

---

### 3. Increase DataLoader Workers

**Location**: train_titanmac_nested.py:877

```python
# Current
num_workers=2

# Recommended
num_workers=4
prefetch_factor=4
```

**Expected Gain**: 3-5% (less I/O stall)

---

## Tier 2: Medium Impact, Medium Effort

### 4. Per-Module Learning Rate Scaling

**Inspiration**: modded-nanogpt uses 75x LR for embeddings, 2x for projections

**Current**: DeepNestedOptimizer controller learns multipliers, but starting ratios are 1:1

**Solution**: Initialize controller with biased outputs or add manual multipliers:
```python
LR_MULTIPLIERS = {
    'embed': 10.0,    # Token embeddings learn faster
    'output': 2.0,    # Output projection
    'default': 1.0,
}
```

**Expected Gain**: 10-20% faster convergence (fewer steps to same loss)

---

### 5. Async DataLoader with Shard Preloading

**Location**: data/loader.py

**Solution**: Background thread preloads next batch while GPU processes current:
```python
import threading
from queue import Queue

class AsyncDataPreloader:
    def __init__(self, loader, prefetch=2):
        self.queue = Queue(maxsize=prefetch)
        self.loader = loader
        threading.Thread(target=self._preload, daemon=True).start()

    def _preload(self):
        for batch in self.loader:
            self.queue.put(batch)
```

**Expected Gain**: 5-10% (eliminates I/O stalls)

---

### 6. Dynamic Momentum Schedule

**Inspiration**: modded-nanogpt warms momentum 0.85 -> 0.95 over 300 steps

**Location**: Would need changes to DeepNestedOptimizer

```python
def get_momentum(step, warmup=300):
    if step < warmup:
        return 0.85 + 0.10 * (step / warmup)
    return 0.95
```

**Expected Gain**: 3-5% faster convergence

---

## Tier 3: High Impact, High Effort

### 7. Polar Express (NOT APPLICABLE)

**Note**: Polar Express replaces Newton-Schulz orthogonalization in Muon optimizer.

**DeepNestedOptimizer does NOT use Newton-Schulz** - it uses AdamW as its base optimizer. Therefore, Polar Express is not applicable here.

If you want to use Polar Express, you would need to:
1. Switch to Muon-based training (train_moe.py)
2. Replace `zeropower_via_newtonschulz5` in optimizers/muon.py

---

### 8. Flash Attention 3

**Status**: User reports Flash Attention won't work on their GPU

**If Available**: Would provide 20-40% attention speedup

**Alternative**: Current BlockSparseAttention with window_size=512 is already O(T*w) instead of O(T^2)

---

### 9. FP8 Linear Operations

**Requirement**: H100/H200 (Hopper architecture) only

**Impact**: 30-50% faster matmuls

**Implementation**:
```python
# Requires torch 2.1+ and Hopper GPU
x_fp8 = x.to(torch.float8_e4m3fn)
w_fp8 = w.to(torch.float8_e4m3fn)
out = torch._scaled_mm(x_fp8, w_fp8.t(), out_dtype=torch.bfloat16)
```

---

### 10. Custom Triton Kernels

**Inspiration**: modded-nanogpt has XXT_kernel, ba_plus_cAA_kernel

**Use Case**: Fused symmetric matmul operations in optimizer

**Effort**: High - requires Triton expertise

**Expected Gain**: 10-20% optimizer speedup

---

## Architectural Limitations (Cannot Fix Without Redesign)

### Neural Memory torch.autograd.grad()

**Location**: titans_core/memory/neural_memory.py:373

```python
grads = torch.autograd.grad(
    loss,
    self.memory_mlp.parameters(),
    retain_graph=True,  # REQUIRED by design
    create_graph=False,
)
```

**Impact**:
- Prevents CUDA graphs
- Prevents `fullgraph=True` in torch.compile (no workaround - see below)
- Doubles peak memory for activations
- Requires `donated_buffer=False`

**Why fullgraph=True Cannot Work**:
1. `torch.autograd.grad()` is explicitly marked non-traceable by dynamo
2. `@torch._dynamo.disable` creates a graph break
3. `fullgraph=True` prohibits ANY graph breaks
4. These are mutually exclusive - no workaround exists

**Why It Exists**: Neural memory's test-time gradient update is core to the Titans architecture. Removing it would fundamentally change the model.

**Workarounds** (already implemented):
```python
# 1. In neural_memory.py: Decorate update() to allow graph break
@torch._dynamo.disable
def update(self, x, theta_t=None, return_stats=False):
    ...  # Contains torch.autograd.grad()

# 2. In training script: Disable donated buffers
torch._functorch.config.donated_buffer = False

# 3. Use dynamic=False (fullgraph=True is NOT compatible)
model = torch.compile(model, mode='reduce-overhead', dynamic=False)
```

---

## Quick Reference: CLI Flags for Speed

```bash
# Maximum speed (after warmup)
python train_titanmac_nested.py \
    --speedy \
    --eval_every 100 \
    --eval_steps 30 \
    --steps 600

# Debug/profile run
python train_titanmac_nested.py \
    --config debug \
    --steps 100 \
    --speedy
```

---

## Expected Cumulative Gains

If all Tier 1 + Tier 2 optimizations are implemented:

| Current | After Tier 1 | After Tier 2 | Notes |
|---------|--------------|--------------|-------|
| 3.5-3.8 it/s | 4.5-5.0 it/s | 5.5-6.5 it/s | Rough estimates |

**Primary bottleneck remaining**: Neural memory's `retain_graph=True` which is architectural.

---

## Files Reference

| File | Hot Functions |
|------|---------------|
| train_titanmac_nested.py | Training loop, eval loop |
| deep_nested_optimizer.py | `_fused_clip_grad_norm_`, `_compute_group_stats`, `preprocess_gradient` |
| nested_controller.py | `forward()` (compiled via use_compile flag) |
| neural_memory.py | `update()` (contains retain_graph=True) |
