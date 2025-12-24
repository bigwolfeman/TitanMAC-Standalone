# TitanMAC + DeepNestedOptimizer GPU Profiling Report

**Date**: December 20, 2025
**Hardware**: NVIDIA GeForce RTX 5090
**PyTorch**: 2.9.1+cu128
**CUDA**: 12.8

## Executive Summary

Comprehensive GPU profiling of TitanMAC with DeepNestedOptimizer revealed severe GPU underutilization. The training loop was **CPU-bound** with only 11% GPU efficiency due to Python dispatch overhead and excessive small kernel launches.

### Key Metrics (Before Optimization)

| Metric | Value | Assessment |
|--------|-------|------------|
| CUDA time per step | 4.4ms | GPU underutilized |
| CPU time per step | 39.9ms | Python overhead dominant |
| GPU efficiency | ~11% | **CRITICAL** |
| Kernel launches/step | 1,421 | Too many small kernels |
| Avg kernel duration | 3.1µs | Not saturating SMs |

---

## Root Cause Analysis

### The Dispatch Overhead Problem

```
CPU (Python)                          GPU
    |                                   |
    |-- dispatch kernel 1 ------------> | (3µs work)
    |<-- wait for completion -----------|
    |-- Python loop iteration           |
    |-- dispatch kernel 2 ------------> | (3µs work)
    |<-- wait for completion -----------|
    ... repeat 1400 times per step ...
```

The GPU completes each kernel in ~3µs, then idles while Python dispatches the next one. This is not a "data on CPU" problem - all tensors are on GPU. It's a **kernel launch overhead** problem.

---

## Top Bottlenecks Identified

### 1. CRITICAL: `_compute_group_stats()` - 65% of CUDA Time

**Location**: `titans_core/opt/deep_nested_optimizer.py`, lines 692-726

**Problem**: Per-parameter Python iteration causing 1660 small kernel launches:
```python
# BEFORE: O(n) kernel launches
for param in group['params']:
    param_norms_sq.append(param.pow(2).sum())  # New kernel each iteration
    if param.grad is not None:
        grad_norms_sq.append(param.grad.pow(2).sum())  # Another kernel
```

**Impact**:
- 1660 calls to `aten::sum` (3µs each = 5ms total)
- 570+ small kernel launches
- Python iteration overhead between kernels
- Implicit GPU synchronization

### 2. HIGH: `clip_grad_norm_` - 23% of CUDA Time

**Location**: Called from `DeepNestedOptimizer.step()`

**Problem**: Default PyTorch implementation iterates over parameters individually.

### 3. MEDIUM: Controller Forward - 18% of CUDA Time

**Location**: `titans_core/opt/nested_controller.py`

**Problem**: 4,385-parameter network taking 8ms for inference on [2,3] input. Small tensor operations don't saturate GPU compute units.

### 4. MEDIUM: Excessive `aten::copy_` - 3010 calls

**Problem**: Tensor copying from:
- Non-contiguous tensor access patterns
- Slice assignments creating implicit copies
- Lack of pre-allocated buffers

### 5. LOW: Neural Memory `autograd.grad()` in Forward

**Location**: `titans_core/memory/neural_memory.py`, line 373

**Problem**: Breaks AMP compatibility, requires `retain_graph=True`.

---

## Memory Analysis

### VRAM Usage (2x vs Muon+AdamW Baseline)

| Component | Memory | Notes |
|-----------|--------|-------|
| Model parameters | ~16MB | 4M params @ fp32 |
| CMS momentum buffers | ~48MB | 3 levels per parameter |
| Controller + MomentumMLP | ~2MB | Small but duplicated |
| Gradient graph retention | ~16MB | For neural memory updates |
| AdamW state (m, v) | ~32MB | Per-parameter buffers |
| **Total** | ~114MB | 2x baseline |

### Memory Bandwidth

| Operation | Bytes | Utilization |
|-----------|-------|-------------|
| `aten::mm` (GEMM) | 729 MB | Good - compute bound |
| `aten::bmm` | 659 MB | Good - compute bound |
| `aten::copy_` | ~800 MB | Poor - pure memory movement |
| `aten::sum` | ~191 KB | Very low - reduction overhead |

---

## Kernel-Level Breakdown

Top 15 CUDA kernels by time:

```
Kernel              Time (ms)   Calls    Avg (µs)   Notes
─────────────────────────────────────────────────────────
aten::mm            10.4        570      18         Matrix multiply
aten::bmm            6.3       1200       5         Batched matmul
aten::sum            5.0       1660       3         Reductions (bottleneck!)
aten::copy_          4.5       3010       1.5       Memory copies
aten::addmm          3.6        240      15         Fused add + matmul
aten::matmul         3.5        510       7         General matmul
aten::clone          2.1       1130       2         Tensor cloning
aten::slice_backward 2.0        870       2         Slice gradient
aten::pow            1.6       1460       1         Elementwise power
aten::mul            1.6       1280       1         Elementwise multiply
```

**Key observation**: `aten::sum` has 1660 calls but only 5ms total - each call is too small to efficiently use the GPU.

---

## Profiling Methodology

### Tools Used

1. **PyTorch Profiler** (`torch.profiler`)
   - CPU and CUDA activity tracing
   - Memory allocation tracking
   - FLOPS estimation
   - Chrome trace export

2. **NVIDIA Nsight Compute** (attempted)
   - Requires elevated permissions
   - Would provide SM occupancy, memory throughput

### Profiling Scripts Created

| Script | Purpose |
|--------|---------|
| `profile_titanmac_torch.py` | PyTorch native profiler with annotations |
| `profile_titanmac_nested.py` | Basic timing script for ncu |

### Running the Profiler

```bash
# Standard profiling
python profile_titanmac_torch.py \
    --num-steps 10 \
    --warmup-steps 3 \
    --batch-size 4 \
    --seq-length 256 \
    --d-model 256 \
    --momentum-layers 4 \
    --controller-layers 5 \
    --output-dir ./profiling_results \
    --trace

# View trace in Chrome
# Open chrome://tracing and load profiling_results/trace.json
```

---

## Comparison: AdamW Mode vs CMS Mode

| Metric | AdamW Mode | CMS Mode | Delta |
|--------|------------|----------|-------|
| CUDA time/step | 4.4ms | 4.5ms | +2% |
| CPU time/step | 39.9ms | 40.1ms | +0.5% |
| Peak memory | 0.42 GB | 0.43 GB | +2% |
| Throughput | 6.04 step/s | 5.98 step/s | -1% |

**Conclusion**: CMS mode adds minimal overhead. The bottleneck is `_compute_group_stats`, not the CMS logic.

---

## References

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Understanding CUDA Kernel Launch Overhead](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-the-current-device-to-avoid-multithreading-bugs/)

---

**Next**: See `PERFORMANCE_OPTIMIZATIONS.md` for the fixes implemented.
