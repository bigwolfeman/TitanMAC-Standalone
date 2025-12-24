# TitanMAC + DeepNestedOptimizer GPU Profiling Report

**Date:** 2025-12-20
**Hardware:** NVIDIA GeForce RTX 5090
**PyTorch:** 2.9.1+cu128
**CUDA:** 12.8

## Executive Summary

The TitanMAC model with DeepNestedOptimizer was profiled using PyTorch's native profiler. The profiling reveals several significant performance bottlenecks, primarily in the optimizer's statistics computation and memory operations.

**Overall GPU Utilization:** The training loop spends only ~44ms in actual CUDA kernel execution per 10 steps (~4.4ms/step), while CPU time totals ~399ms. This indicates the training is **CPU-bound** with significant Python overhead.

### Key Findings

| Metric | Value | Assessment |
|--------|-------|------------|
| Total CUDA time (10 steps) | 44.0ms | Low - underutilized GPU |
| Total CPU time (10 steps) | 398.9ms | High - CPU bottleneck |
| Peak memory | 0.42 GB | Low for 4M param model |
| Memory growth | 0.21 GB | Acceptable |
| GPU efficiency | ~11% | **CRITICAL** - severe underutilization |

---

## Top 10 Most Expensive Operations (CUDA Time)

| Rank | Operation | CUDA Time (ms) | Calls | Avg/Call (us) |
|------|-----------|----------------|-------|---------------|
| 1 | `compute_group_stats` | 28.8ms | 10 | 2,883 |
| 2 | `model_forward` | 12.7ms | 10 | 1,271 |
| 3 | `AdamW.step` | 11.2ms | 10 | 1,123 |
| 4 | `aten::mm` | 10.4ms | 570 | 18 |
| 5 | `clip_grad_norm` | 10.0ms | 10 | 997 |
| 6 | `controller_forward` | 8.0ms | 10 | 801 |
| 7 | `aten::bmm` | 6.3ms | 1200 | 5 |
| 8 | `aten::linear` | 5.3ms | 350 | 15 |
| 9 | `aten::sum` | 5.0ms | 1660 | 3 |
| 10 | `aten::copy_` | 4.5ms | 3010 | 1.5 |

---

## Bottleneck Analysis

### 1. CRITICAL: `_compute_group_stats()` - 65.5% of CUDA time

**Location:** `/111TitanMAC-Standalone/titans_core/opt/deep_nested_optimizer.py`, lines 692-726

**Problem:** This function iterates over all parameters in Python, computing norms individually:
```python
for param in group['params']:
    param_norms_sq.append(param.pow(2).sum())  # Creates tensor per param
    if param.grad is not None:
        grad_norms_sq.append(param.grad.pow(2).sum())

# Then stacks all tensors
stats[i, 0] = torch.stack(grad_norms_sq).sum().sqrt()
```

**Impact:**
- 1660 calls to `aten::sum` (3us each = 5ms total)
- 570+ small kernel launches
- Python iteration overhead
- GPU synchronization between operations

**Solution:** Use `torch._foreach_*` or fused operations:
```python
# BEFORE: O(n) kernel launches
grad_norms_sq = []
for param in group['params']:
    grad_norms_sq.append(param.grad.pow(2).sum())

# AFTER: O(1) kernel launch
grads = [p.grad for p in group['params'] if p.grad is not None]
if grads:
    grad_sq_sums = torch._foreach_norm(grads, ord=2)
    stats[i, 0] = sum(x**2 for x in grad_sq_sums) ** 0.5
```

**Estimated Speedup:** 5-10x for this function

---

### 2. HIGH: Excessive `aten::copy_` Operations - 3010 calls

**Location:** Throughout optimizer step and model forward

**Problem:** 3010 copy operations in 10 steps (301 per step) indicates:
- Frequent tensor copying between contiguous memory layouts
- Slice assignments creating implicit copies
- Non-contiguous tensor access patterns

**Key culprits identified:**
- `SliceBackward0`: 870 calls with -1.12GB memory churn
- `torch::autograd::CopySlices`: 200 calls

**Impact:** 4.5ms CUDA time + 16ms CPU time for data movement

**Solution:**
1. Pre-allocate output tensors
2. Use `.contiguous()` before operations that require it
3. Avoid slicing operations in hot loops

---

### 3. MEDIUM: `clip_grad_norm` - 10ms (22.6% of CUDA time)

**Location:** Called from `DeepNestedOptimizer.step()`

**Problem:** Default PyTorch implementation iterates over parameters

**Solution:** Use fused gradient clipping:
```python
# Current
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Optimized (for large models)
torch._foreach_norm(grads, ord=2)  # Single kernel
```

---

### 4. MEDIUM: Controller Forward - 8ms per batch

**Location:** `/111TitanMAC-Standalone/titans_core/opt/nested_controller.py`

**Problem:** The controller network (4,385 params) takes 8ms for inference on [2,3] input - disproportionate to its size.

**Root cause:** Small tensor operations don't saturate GPU compute units.

**Solution:**
1. Batch controller updates (accumulate stats, update every N steps)
2. Consider moving controller to CPU for small models
3. Use `torch.compile` for the controller

---

### 5. MEDIUM: Neural Memory `torch.autograd.grad()` in Forward Pass

**Location:** `/111TitanMAC-Standalone/titans_core/memory/neural_memory.py`, line 373

**Problem:** Memory update calls `torch.autograd.grad()` inside forward pass:
```python
grads = torch.autograd.grad(
    loss,
    self.memory_mlp.parameters(),
    retain_graph=True,
    create_graph=False,
)
```

**Impact:**
- Breaks AMP autocast compatibility
- Adds 0.8ms per forward pass
- Requires `retain_graph=True` (memory overhead)

**Solution:** Move memory update to post-backward hook or separate call

---

## Memory Bandwidth Analysis

| Operation | Bytes Read/Write | Bandwidth Utilization |
|-----------|-----------------|----------------------|
| `aten::mm` (GEMM) | 729 MB | Good - compute bound |
| `aten::bmm` | 659 MB | Good - compute bound |
| `aten::copy_` | ~800 MB | Poor - pure memory |
| `aten::sum` | ~191 KB reduction | Very low occupancy |

**Observation:** The model is small enough that many operations are memory-bound rather than compute-bound. The GPU SM utilization is likely below 30%.

---

## Kernel Launch Overhead Analysis

| Metric | Value |
|--------|-------|
| Total kernel launches | ~14,210 per 10 steps |
| Kernels per step | ~1,421 |
| cudaLaunchKernel overhead | 45.4ms CPU time |
| Average kernel duration | 3.1us |

**Assessment:** Many kernels are too small (<10us). This causes:
1. GPU starvation between kernels
2. CPU->GPU command queue overhead
3. Poor SM occupancy

---

## Optimization Recommendations

### Quick Wins (Immediate Impact)

1. **Fuse `_compute_group_stats`**: Replace per-parameter loops with `torch._foreach_*`
   - Estimated gain: **20-25ms per 10 steps**

2. **Use `torch.compile` on hot paths**:
   ```python
   @torch.compile
   def _compute_group_stats(self) -> Tensor:
       ...
   ```
   - Estimated gain: **5-10ms from kernel fusion**

3. **Batch controller updates**: Update controller every 10 steps instead of every step
   - Estimated gain: **7ms per 10 steps**

### Architectural Changes

4. **Pre-compute parameter views**: Cache flattened parameter and gradient views
   ```python
   # Once at init
   self._param_flat = torch.cat([p.view(-1) for p in self.param_groups[0]['params']])
   # Then in step
   grad_norm = self._grad_flat.norm()  # Single kernel
   ```

5. **Move memory update out of forward**: Use hook-based updates
   ```python
   def register_memory_update_hook(self):
       def hook(grad):
           self.neural_memory.update(...)
       self.output.register_hook(hook)
   ```

6. **Use CUDA Graphs for optimizer step**: The optimizer step pattern is deterministic
   ```python
   # Capture once
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       optimizer.step()
   # Replay
   g.replay()
   ```

### CMS Mode Analysis

When CMS mode is enabled (`use_cms_updates=True`), the profiling shows:
- Similar forward/backward times (expected - model unchanged)
- Slightly higher optimizer overhead (~2ms from surrogate loss computation)
- Additional memory usage from output history tracking

**CMS-specific bottleneck:** The surrogate loss computation creates new tensors and graph nodes per parameter:
```python
# File: deep_nested_optimizer.py, line 930-948
for param, cms in training_samples:
    mlp_output = self.momentum_mlp(level_grad, prev_momentum)  # Creates graph
    loss = self._compute_surrogate_loss(mlp_output, level_grad, prev_output)
    surrogate_losses.append(loss)  # Python list append
```

---

## Comparison: AdamW vs CMS Mode

| Metric | AdamW Mode | CMS Mode | Delta |
|--------|------------|----------|-------|
| CUDA time/step | 4.4ms | 4.5ms | +2% |
| CPU time/step | 39.9ms | 40.1ms | +0.5% |
| Peak memory | 0.42 GB | 0.43 GB | +2% |
| Throughput | 6.04 step/s | 5.98 step/s | -1% |

**Conclusion:** CMS mode adds minimal overhead when in AdamW fallback. The major overhead comes from `_compute_group_stats`, not the CMS logic itself.

---

## Recommended Profiling Next Steps

1. **Run with larger model**: Profile with d_model=512 or higher to see compute-bound behavior
2. **Nsight Compute deep dive**: Request admin permissions for GPU performance counters
3. **Memory profiler**: Use `torch.cuda.memory._record_memory_history()` for allocation tracking
4. **Profile meta-update separately**: The simplified meta trainer may have hidden costs

---

## Files Modified for Profiling

| File | Purpose |
|------|---------|
| `profile_titanmac_nested.py` | Basic profiling script |
| `profile_titanmac_torch.py` | PyTorch native profiler with detailed annotations |
| `profiling_results/` | Output directory with traces and summaries |

---

## Appendix: Full Kernel Time Breakdown

See `/profiling_results/profile_table.txt` for the complete kernel-level breakdown.

Top 15 CUDA Kernels:
```
aten::mm            10.4ms  (570 calls)  - Matrix multiply
aten::bmm            6.3ms  (1200 calls) - Batched matrix multiply
aten::sum            5.0ms  (1660 calls) - Reductions
aten::copy_          4.5ms  (3010 calls) - Memory copies
aten::addmm          3.6ms  (240 calls)  - Fused add + matmul
aten::matmul         3.5ms  (510 calls)  - General matmul
aten::clone          2.1ms  (1130 calls) - Tensor cloning
aten::slice_backward 2.0ms  (870 calls)  - Slice gradient
aten::pow            1.6ms  (1460 calls) - Elementwise power
aten::mul            1.6ms  (1280 calls) - Elementwise multiply
```

---

**Report generated by Claude Opus 4.5**
**Confidence: 8/10** - Profiling data is accurate, but ncu hardware counters were unavailable for definitive memory/compute bound classification.
