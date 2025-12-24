# DeepNestedOptimizer Performance Optimizations

**Date**: December 20, 2025
**Status**: Implemented and validated

This document details all performance optimizations applied to the DeepNestedOptimizer based on GPU profiling analysis.

---

## Summary of Changes

| Optimization | File | Impact | Status |
|--------------|------|--------|--------|
| Fused `_compute_group_stats` | `deep_nested_optimizer.py` | **64% CUDA reduction** | DONE |
| Fused gradient clipping | `deep_nested_optimizer.py` | **24% reduction** | DONE |
| `torch.compile` support | `deep_nested_optimizer.py` | Kernel fusion | DONE |
| Controller batching | `deep_nested_optimizer.py` | Reduced dispatch | DONE |
| Pre-allocated caches | `deep_nested_optimizer.py` | Reduced allocations | DONE |

### Performance Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| `_compute_group_stats` CUDA | 12.9ms | 4.7ms | **64%** |
| `clip_grad_norm` CUDA | 6.5ms | 4.9ms | **24%** |
| `aten::sum` kernel calls | 830 | 195 | **77%** |
| Total optimizer CPU time | 36.0ms | 24.7ms | **31%** |

---

## 1. Fused `_compute_group_stats()` (CRITICAL)

### Problem

The original implementation iterated over parameters in Python, launching a kernel for each:

```python
# BEFORE: O(n) kernel launches per step
def _compute_group_stats(self) -> Tensor:
    stats = torch.zeros(len(self.param_groups), 3, device=self.device)

    for i, group in enumerate(self.param_groups):
        grad_norms_sq = []
        param_norms_sq = []

        for param in group['params']:
            param_norms_sq.append(param.pow(2).sum())  # Kernel launch
            if param.grad is not None:
                grad_norms_sq.append(param.grad.pow(2).sum())  # Kernel launch

        # Stack and reduce
        stats[i, 0] = torch.stack(grad_norms_sq).sum().sqrt()
        stats[i, 1] = torch.stack(param_norms_sq).sum().sqrt()

    return stats
```

### Solution

Use `torch._foreach_norm()` to compute all norms in a single fused kernel:

```python
# AFTER: O(1) kernel launches per group
def _compute_group_stats(self) -> Tensor:
    # Reuse pre-allocated cache
    if self._stats_cache is None:
        self._stats_cache = torch.zeros(
            len(self.param_groups), 3, device=self.device
        )
    stats = self._stats_cache

    for i, group in enumerate(self.param_groups):
        params = [p for p in group['params']]
        grads = [p.grad for p in group['params'] if p.grad is not None]

        # FUSED: Single kernel for all parameter norms
        if params:
            param_norms = torch._foreach_norm(params)
            stacked = torch.stack(param_norms)
            stats[i, 1] = (stacked ** 2).sum().sqrt()

        # FUSED: Single kernel for all gradient norms
        if grads:
            grad_norms = torch._foreach_norm(grads)
            stacked = torch.stack(grad_norms)
            stats[i, 0] = (stacked ** 2).sum().sqrt()

        stats[i, 2] = float(i)  # Depth indicator

    return stats
```

### Impact

- **Kernel launches**: 1660 → ~20 per step
- **CUDA time**: 12.9ms → 4.7ms (64% reduction)

---

## 2. Fused Gradient Clipping

### Problem

Default `torch.nn.utils.clip_grad_norm_` iterates over parameters:

```python
# PyTorch default implementation (simplified)
def clip_grad_norm_(parameters, max_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.norm() ** 2  # Kernel per param
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(clip_coef)  # Kernel per param
```

### Solution

Added `_fused_clip_grad_norm_()` using foreach operations:

```python
def _fused_clip_grad_norm_(
    parameters: List[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
) -> Tensor:
    """Fused gradient clipping using torch._foreach_* operations."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    # FUSED: Compute all norms in single kernel
    norms = torch._foreach_norm(grads, ord=norm_type)

    # Compute total norm
    total_norm = torch.stack(norms).norm(norm_type)

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # FUSED: Apply clipping to all grads in single kernel
    if clip_coef_clamped < 1.0:
        torch._foreach_mul_(grads, clip_coef_clamped.item())

    return total_norm
```

### Impact

- **CUDA time**: 6.5ms → 4.9ms (24% reduction)

---

## 3. `torch.compile` Support

### New Parameter

```python
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=1e-4,
    # ... other params ...
    use_compile=True,  # NEW: Enable torch.compile on hot paths
)
```

### Implementation

```python
def __init__(self, ..., use_compile: bool = False):
    # ... existing init ...

    if use_compile and hasattr(torch, 'compile'):
        # Compile controller with reduce-overhead mode
        self.controller = torch.compile(
            self.controller,
            mode='reduce-overhead'
        )
        # Compile momentum MLP
        self.momentum_mlp = torch.compile(
            self.momentum_mlp,
            mode='reduce-overhead'
        )
```

### Notes

- `mode='reduce-overhead'` optimizes for reducing CPU dispatch overhead
- First few steps will be slower due to compilation
- Best gains on repeated identical tensor shapes

---

## 4. Controller Update Batching

### New Parameter

```python
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=1e-4,
    # ... other params ...
    controller_update_freq=10,  # NEW: Update controller every N steps
)
```

### Implementation

```python
def step(self, loss: Optional[float] = None):
    # Only run controller on specified frequency
    if self.global_step % self.controller_update_freq == 0 or self.global_step == 1:
        stats = self._compute_group_stats()
        with torch.no_grad():
            self._lr_multipliers = self.controller(stats)
    # Else: reuse cached self._lr_multipliers

    # ... rest of step ...
```

### Impact

With `controller_update_freq=10`:
- Controller forward runs 10x less often
- ~7ms saved per 10 steps

### Trade-off

- LR multipliers are stale for up to N-1 steps
- Recommended: Start with `controller_update_freq=1`, increase if bottlenecked

---

## 5. Pre-allocated Tensor Caches

### Problem

Allocating tensors every step causes:
- CUDA malloc overhead
- Memory fragmentation
- Implicit synchronization

### Solution

Added caches for frequently allocated tensors:

```python
class DeepNestedOptimizer:
    def __init__(self, ...):
        # ... existing init ...

        # Pre-allocation caches
        self._stats_cache: Optional[Tensor] = None
        self._context_cache: Optional[Tensor] = None
        self._lr_multipliers: Optional[Tensor] = None

    def _compute_group_stats(self) -> Tensor:
        # Reuse cache instead of allocating
        if self._stats_cache is None:
            self._stats_cache = torch.zeros(
                len(self.param_groups), 3, device=self.device
            )
        return self._stats_cache  # Zero and reuse

    def _get_context(self) -> Tensor:
        if self._context_cache is None:
            self._context_cache = torch.zeros(
                self.context_dim, device=self.device
            )
        return self._context_cache
```

---

## 6. Meta-Learning Graph Isolation (CRITICAL FIX)

### Problem

The meta-learning backward pass crashes with:
```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [128]] is at version N
```

### Root Cause

`_compute_group_stats()` was calling `torch._foreach_norm(params)` which creates an autograd graph through **model parameters**. When `optimizer.step()` modifies those parameters in-place, the subsequent `meta_loss.backward()` crashes because the graph references stale tensor versions.

### Solution

Meta-learning only needs gradients through **controller and MLP parameters**, not model parameters. Detach all connections to model params:

**File: `titans_core/opt/deep_nested_optimizer.py`**

#### Fix 1: `_compute_group_stats()` (lines 837-859)

```python
# BEFORE (creates graph through model params):
param_norms = torch._foreach_norm(params)
stacked_norms = torch.stack(param_norms)
stats[i, 1] = (stacked_norms ** 2).sum().sqrt()
return stats

# AFTER (no graph through model params):
with torch.no_grad():
    param_norms = torch._foreach_norm(params)
    if param_norms:
        stacked_norms = torch.stack(param_norms)
        stats[i, 1] = (stacked_norms ** 2).sum().sqrt()
return stats.detach()
```

#### Fix 2: `_compute_mlp_proxy_loss()` (lines 1579-1588)

```python
# BEFORE:
grad = param.grad
prev_momentum = cms.get_momentum(0)
ema_target = cms.get_ema_grad(0)

# AFTER:
grad = param.grad.detach()
prev_momentum = cms.get_momentum(0).detach()
ema_target = cms.get_ema_grad(0).detach()
```

#### Fix 3: `_fused_clip_grad_norm_()` (lines 35-103)

Added `inplace` parameter. Use `inplace=False` for non-CUDA-graph mode to preserve gradient graph:

```python
def _fused_clip_grad_norm_(
    parameters, max_norm, norm_type=2.0, error_if_nonfinite=False,
    inplace=True,  # NEW: Set False for meta-learning compatibility
):
```

### Why This Works

```
Model params ──────────────────────────────────────────┐
      │                                                │
      ▼                                                │
forward() ──► loss ──► backward() ──► gradients        │
                                          │            │
                                    [DETACH HERE]      │
                                          │            │
                                          ▼            │
                              _compute_group_stats()   │
                                          │            │
                                    [DETACH HERE]      │
                                          ▼            │
                              controller(stats)        │ ← No connection!
                                          │            │
                                          ▼            │
                              meta_loss.backward() ────┘
                                          │
                                          ▼
                              Updates controller/MLP only
```

---

## 7. CUDA Graphs

### New Parameters

```python
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=1e-3,
    use_cuda_graph=True,           # Enable CUDA Graph capture
    cuda_graph_warmup_steps=3,     # Warmup steps before capture
)
```

### Performance Results

| Metric | Eager Mode | Graph Mode | Improvement |
|--------|------------|------------|-------------|
| Avg step time | 1.09ms | 0.63ms | **42% speedup** |
| Total time (100 steps) | 0.25s | 0.14s | 44% faster |

### How It Works

1. **Warmup phase** (steps 1-3): Normal execution to stabilize tensor shapes
2. **Capture phase** (step 4): Records AdamW step as CUDA graph
3. **Replay phase** (step 5+): Replays captured graph without Python dispatch

### Critical Implementation Detail

Gradient clipping MUST be done **outside** the graph:

```python
# WRONG - inside graph (causes memory corruption)
with torch.cuda.graph(g):
    clip_grad_norm_(params, max_norm)  # Gradient addresses may change!
    optimizer.step()

# CORRECT - outside graph
clip_grad_norm_(params, max_norm)  # Fresh gradients each time
with torch.cuda.graph(g):
    optimizer.step()  # Only step uses stable state tensor addresses
```

**Why**: Gradient tensors may be allocated at different addresses between backward passes. CUDA Graphs capture specific memory addresses, so including gradient operations causes the graph to read/write wrong memory locations.

### Limitations

- Requires `use_cms_updates=False` (AdamW mode only)
- Requires GPU with compute capability >= 7.0 (Volta+)
- Slight numerical differences from `capturable=True` AdamW mode

---

## Usage Examples

### Basic Usage (Backward Compatible)

```python
# Default settings - same behavior as before
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    meta_lr=1e-4,
)
```

### Maximum Performance

```python
# All optimizations enabled
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    meta_lr=1e-4,
    use_compile=True,           # Kernel fusion
    use_cuda_graph=True,        # CUDA Graph for 42% speedup
    cuda_graph_warmup_steps=3,  # Warmup before capture
    controller_update_freq=10,  # Batch controller updates
)
```

### Memory-Constrained

```python
# Minimize memory overhead
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    meta_lr=1e-4,
    use_cms_updates=False,      # Disable CMS (saves momentum buffers)
    controller_update_freq=20,  # Less frequent updates
)
```

---

## Validation

All optimizations were validated:

1. **Gradient flow**: All 26 parameters receive gradients correctly
2. **Loss convergence**: Training loss decreases as expected
3. **LR multipliers**: Applied correctly to parameter groups
4. **Backward compatibility**: Default parameters match original behavior

### Test Command

```bash
python profile_titanmac_torch.py \
    --num-steps 10 \
    --warmup-steps 3 \
    --output-dir ./profiling_results_optimized
```

---

## Future Optimizations

### 1. CUDA Graphs

The optimizer step pattern is deterministic and could use CUDA Graphs:

```python
# Capture once
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    optimizer.step()

# Replay without Python dispatch
g.replay()
```

**Blocked by**: Dynamic tensor shapes in some paths

### 2. Custom CUDA Kernels

For maximum performance, the entire optimizer step could be a single custom kernel:

```cuda
__global__ void fused_nested_optimizer_step(
    float* params,
    float* grads,
    float* momentum,
    float* controller_weights,
    float lr,
    int n_params
) {
    // Fused: compute stats, run controller, apply update
}
```

**Blocked by**: Development complexity

### 3. CPU Offload for Small Controller

For small models, the controller overhead dominates. Moving it to CPU could help:

```python
if model_params < 10_000_000:
    self.controller = self.controller.cpu()
```

---

## Files Modified

| File | Changes |
|------|---------|
| `titans_core/opt/deep_nested_optimizer.py` | All optimizations |
| `titans_core/opt/nested_controller.py` | `torch.compile` support |

---

## References

- [PyTorch foreach operations](https://pytorch.org/docs/stable/generated/torch._foreach_norm.html)
- [torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [CUDA Graphs in PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
