# Triton Kernel Implementation Plan

**Date**: December 20, 2025
**Status**: Implementation Ready
**Goal**: Fuse hot paths into optimized Triton kernels

---

## Profiling Results Summary

From PyTorch profiler on TitanMAC training (d_model=512, 6 layers, batch=4, seq=512):

### CUDA Time Breakdown

| Operation | CUDA Time | % | Priority |
|-----------|-----------|---|----------|
| `aten::mm` (matmul) | 79.28ms | 34.1% | Tensor cores already used |
| FMHA backward | 36.95ms | 15.9% | Flash Attention (optimized) |
| AdamW.step | 39.60ms | 17.0% | **FUSE OPPORTUNITY** |
| `aten::addmm` | 26.02ms | 11.2% | Part of linear layers |
| log_softmax_backward | 12.38ms | 5.3% | Cross-entropy backward |
| FMHA forward | 10.75ms | 4.6% | Flash Attention (optimized) |
| SiLU forward/backward | ~2.4ms | 1.0% | **FUSE INTO MLP** |

### CRITICAL CPU OVERHEAD

| Operation | CPU Time | % | Issue |
|-----------|----------|---|-------|
| `aten::item()` | 90.47ms | 40.0% | **CUDA SYNC EVERY STEP** |
| `cudaStreamSynchronize` | 87.64ms | 38.7% | From .item() calls |

**ROOT CAUSE**: NaN guards use `.item()` which syncs GPU on every step.

---

## Priority Kernels to Implement

### Priority 1: Eliminate CPU Sync (IMMEDIATE - NO TRITON NEEDED)

**Problem**: Every training step syncs CPU-GPU 3120 times due to `.item()` calls.

**Files affected**:
- `nested_controller.py:102,127` - NaN guard
- `deep_nested_optimizer.py:861,1291` - NaN guards
- `neural_memory.py:438-440` - Stats return

**Solution**: Use GPU-side NaN checking without sync:

```python
# BEFORE (syncs GPU)
if torch.isnan(stats).any().item():  # .item() syncs!
    return defaults

# AFTER (no sync)
nan_mask = torch.isnan(stats).any()
# Use torch.where for branchless computation
result = torch.where(nan_mask, defaults, computed_result)
```

**Expected Gain**: 40-50% wall-clock speedup (removes 90ms of sync per training batch)

---

### Priority 2: Fused Memory MLP Update Kernel

**Current Operation Sequence** (neural_memory.update()):
```
1. k = W_K @ x          # Linear projection [N, D]
2. v = W_V @ x          # Linear projection [N, D]
3. z1 = k @ W1.T + B1   # Memory MLP layer 1
4. h = silu(z1)         # Activation
5. y = h @ W2.T + B2    # Memory MLP layer 2
6. loss = MSE(y, v)     # Loss computation
7. grads = autograd.grad(loss, [W1,B1,W2,B2])  # THE BOTTLENECK
8. clip grads           # Gradient clipping
9. S = η*S - θ*g        # Momentum update
10. W = (1-α)*W + S     # Weight update
```

**Kernel Fusion Strategy**:

Fuse into 2 kernels:
1. **Forward + Activation kernel**: Steps 3-6
2. **Backward + Update kernel**: Steps 7-10 (manual gradients, no autograd)

**Fused Kernel A: memory_mlp_forward_loss**

```python
@triton.jit
def memory_mlp_forward_loss_kernel(
    # Inputs
    k_ptr,      # [N, D_in]
    v_ptr,      # [N, D_in] target
    W1_ptr,     # [H, D_in]
    B1_ptr,     # [H]
    W2_ptr,     # [D_out, H]
    B2_ptr,     # [D_out]

    # Outputs
    z1_ptr,     # [N, H] - save for backward
    h_ptr,      # [N, H] - save for backward
    y_ptr,      # [N, D_out] - save for backward
    loss_ptr,   # [1] - scalar loss

    # Dimensions
    N, D_in, H, D_out,

    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fused forward pass through 2-layer MLP + MSE loss.

    Saves intermediate activations for backward pass.
    Computes:
        z1 = k @ W1.T + B1
        h = silu(z1)
        y = h @ W2.T + B2
        loss = MSE(y, v)
    """
    # ... implementation
```

**Fused Kernel B: memory_mlp_backward_update**

```python
@triton.jit
def memory_mlp_backward_update_kernel(
    # Saved activations (read-only)
    k_ptr, v_ptr, z1_ptr, h_ptr, y_ptr,

    # Weights (read-write)
    W1_ptr, B1_ptr, W2_ptr, B2_ptr,

    # Momentum buffers (read-write)
    S_W1_ptr, S_B1_ptr, S_W2_ptr, S_B2_ptr,

    # Hyperparameters
    alpha_t,      # forget gate
    eta_t,        # decay gate
    theta,        # learning rate
    max_grad_norm,

    # Dimensions
    N, D_in, H, D_out,

    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused backward pass + gradient clipping + momentum + weight update.

    All in one kernel - no autograd.grad() needed!

    Computes:
        d_y = (2/N/D) * (y - v)
        d_W2 = d_y.T @ h
        d_B2 = sum(d_y)
        d_h = d_y @ W2
        d_z1 = d_h * silu'(z1)
        d_W1 = d_z1.T @ k
        d_B1 = sum(d_z1)

        grad_norm = sqrt(sum(d**2 for d in [dW1,dB1,dW2,dB2]))
        scale = min(1, max_grad_norm / grad_norm)

        S = η*S - θ*grad*scale
        W = (1-α)*W + S
    """
    # ... implementation
```

**Expected Gain**: 2-3x speedup for memory update (eliminates graph break + kernel launch overhead)

---

### Priority 3: Fused AdamW Step

**Current**: AdamW takes 17% of CUDA time with multiple kernel launches:
- Copy gradients
- Compute m = β1*m + (1-β1)*g
- Compute v = β2*v + (1-β2)*g²
- Compute m_hat, v_hat (bias correction)
- Compute update = m_hat / (sqrt(v_hat) + ε)
- Apply weight decay: w -= λ*w
- Apply update: w -= lr*update

**Fused Kernel**: Single kernel for all AdamW operations per parameter group

```python
@triton.jit
def fused_adamw_kernel(
    # Parameters (read-write)
    param_ptr,
    grad_ptr,

    # State (read-write)
    exp_avg_ptr,      # m (first moment)
    exp_avg_sq_ptr,   # v (second moment)

    # Hyperparameters
    lr, beta1, beta2, eps, weight_decay,
    step,  # For bias correction

    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused AdamW update in single kernel.

    For each element:
        m = β1*m + (1-β1)*g
        v = β2*v + (1-β2)*g²
        m_hat = m / (1 - β1^step)
        v_hat = v / (1 - β2^step)
        param -= lr * (m_hat / (sqrt(v_hat) + ε) + λ*param)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load
    param = tl.load(param_ptr + offs, mask=mask)
    grad = tl.load(grad_ptr + offs, mask=mask)
    m = tl.load(exp_avg_ptr + offs, mask=mask)
    v = tl.load(exp_avg_sq_ptr + offs, mask=mask)

    # Update moments
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad

    # Bias correction
    bias_correction1 = 1 - tl.exp(step * tl.log(beta1))  # 1 - β1^step
    bias_correction2 = 1 - tl.exp(step * tl.log(beta2))  # 1 - β2^step
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # Update with weight decay
    update = m_hat / (tl.sqrt(v_hat) + eps) + weight_decay * param
    param = param - lr * update

    # Store
    tl.store(param_ptr + offs, param, mask=mask)
    tl.store(exp_avg_ptr + offs, m, mask=mask)
    tl.store(exp_avg_sq_ptr + offs, v, mask=mask)
```

**Expected Gain**: 30-50% speedup for optimizer step

---

### Priority 4: Fused LayerNorm + Linear

**Current**: Separate kernels for LayerNorm and following Linear layer.

**Fusion Opportunity**: Combine into single kernel to reduce memory traffic.

```python
@triton.jit
def fused_layernorm_linear_kernel(
    x_ptr,          # [N, D_in]
    weight_ptr,     # [D_in] - LN weight
    bias_ptr,       # [D_in] - LN bias
    W_ptr,          # [D_out, D_in] - Linear weight
    out_ptr,        # [N, D_out]
    ...
):
    """
    Fused: LayerNorm(x) @ W.T

    Eliminates intermediate tensor for normalized x.
    """
```

**Expected Gain**: 10-20% for transformer blocks

---

## Implementation Order

| Phase | Kernel | Effort | Impact | Priority |
|-------|--------|--------|--------|----------|
| 0 | Eliminate .item() syncs | 1 hour | 40-50% | **IMMEDIATE** |
| 1 | memory_mlp_backward_update | 2 days | 20-30% | HIGH |
| 2 | fused_adamw | 1 day | 15-20% | HIGH |
| 3 | memory_mlp_forward_loss | 1 day | 10-15% | MEDIUM |
| 4 | fused_layernorm_linear | 2 days | 10-15% | LOW |

---

## Phase 0: Eliminate CPU Syncs (Do This NOW)

### Files to Modify

**1. nested_controller.py** (lines 100-104, 125-129):

```python
# BEFORE
if torch.isnan(stats).any() or torch.isinf(stats).any():
    nan_count = (torch.isnan(stats) | torch.isinf(stats)).sum().item()  # SYNC!
    print(f"...")
    return torch.ones(...)

# AFTER
nan_mask = torch.isnan(stats) | torch.isinf(stats)
has_nan = nan_mask.any()  # Keep as tensor, no .item()

# Use torch.where for branchless
default = torch.ones(self.n_groups, device=stats.device, dtype=stats.dtype)

# Compute normal result
raw_output = self.net(stats)
normalized = torch.sigmoid(raw_output.squeeze(-1))
multipliers = self.min_lr_mult + normalized * (self.max_lr_mult - self.min_lr_mult)

# Branchless select - no CPU sync!
result = torch.where(has_nan.unsqueeze(-1).expand_as(multipliers), default, multipliers)
return result
```

**2. deep_nested_optimizer.py** (lines 858-868):

```python
# BEFORE
nan_mask = torch.isnan(stats) | torch.isinf(stats)
if nan_mask.any():
    nan_count = nan_mask.sum().item()  # SYNC!
    print(f"...")

# AFTER (branchless)
nan_mask = torch.isnan(stats) | torch.isinf(stats)
defaults = torch.tensor([1.0, 1.0, 0.5], device=self.device).unsqueeze(0).expand_as(stats)
stats = torch.where(nan_mask, defaults, stats)  # Branchless, no sync
```

**3. neural_memory.py** (lines 399-409):

```python
# BEFORE
if torch.isnan(self._flat_grad_cache).any() or torch.isinf(self._flat_grad_cache).any():
    if return_stats:
        return {...}
    return loss

# AFTER (branchless with flag)
has_nan = (torch.isnan(self._flat_grad_cache) | torch.isinf(self._flat_grad_cache)).any()
# Continue with computation, use has_nan to mask final update
# (implementation depends on exact semantics needed)
```

---

## Detailed Kernel Algorithms

### Memory MLP Backward + Update

**Input shapes** (typical):
- N = batch * seq = 4 * 512 = 2048
- D = d_memory = 512
- H = hidden_dim = 512

**Algorithm**:

```
PHASE 1: Gradient Computation

    # Output gradient
    d_y = (2 / (N * D)) * (y - v)   # [N, D]

    # Layer 2 backward
    d_W2 = d_y.T @ h                 # [D, H] = [D, N] @ [N, H]
    d_B2 = sum(d_y, axis=0)          # [D]
    d_h = d_y @ W2                   # [N, H] = [N, D] @ [D, H]

    # SiLU backward
    σ = sigmoid(z1)
    silu_grad = σ * (1 + z1 * (1 - σ))
    d_z1 = d_h * silu_grad           # [N, H]

    # Layer 1 backward
    d_W1 = d_z1.T @ k                # [H, D] = [H, N] @ [N, D]
    d_B1 = sum(d_z1, axis=0)         # [H]

PHASE 2: Gradient Clipping

    # Compute norm across all gradients
    norm_sq = sum(d_W1**2) + sum(d_B1**2) + sum(d_W2**2) + sum(d_B2**2)
    grad_norm = sqrt(norm_sq)
    scale = min(1.0, max_grad_norm / grad_norm)

    # Apply scale
    d_W1 *= scale
    d_B1 *= scale
    d_W2 *= scale
    d_B2 *= scale

PHASE 3: Momentum + Weight Update

    # Momentum: S = η*S - θ*g
    S_W1 = η * S_W1 - θ * d_W1
    S_B1 = η * S_B1 - θ * d_B1
    S_W2 = η * S_W2 - θ * d_W2
    S_B2 = η * S_B2 - θ * d_B2

    # Weight: W = (1-α)*W + S
    W1 = (1 - α) * W1 + S_W1
    B1 = (1 - α) * B1 + S_B1
    W2 = (1 - α) * W2 + S_W2
    B2 = (1 - α) * B2 + S_B2
```

**Tiling Strategy**:

The key operations are matmul reductions (d_W2, d_W1) which need to accumulate over N.

```
Grid: (D // BLOCK_D, H // BLOCK_H) for d_W2 computation
      (H // BLOCK_H, D // BLOCK_D) for d_W1 computation

For each block:
    1. Load tile of d_y/d_z1 [BLOCK_N, BLOCK_D/H]
    2. Load tile of h/k [BLOCK_N, BLOCK_H/D]
    3. Compute local d_W partial sum
    4. Atomic add to global d_W accumulator
    5. Repeat for all N tiles
```

**Memory Access Pattern**:

```
Read:
    - k: [N, D] - 2048 * 512 * 4 = 4MB
    - v: [N, D] - 4MB
    - z1, h, y: 3 * 4MB = 12MB
    - W1, W2: 2 * 512 * 512 * 4 = 2MB
    Total read: ~24MB

Write:
    - W1, B1, W2, B2: 2 * 512 * 512 * 4 + 2 * 512 * 4 = 2MB
    - S_W1, S_B1, S_W2, S_B2: 2MB
    Total write: ~4MB

Memory bandwidth: ~28MB per update
At 900 GB/s (5090): 0.03ms theoretical minimum
Current: ~5-10ms
Speedup potential: 100-300x (compute bound, not memory bound)
```

---

## File Structure

```
titans_core/
├── kernels/
│   ├── __init__.py
│   ├── memory_mlp.py       # Fused memory MLP kernels
│   ├── adamw.py            # Fused AdamW kernel
│   ├── layernorm_linear.py # Fused LN + Linear
│   └── utils.py            # Shared utilities
├── memory/
│   ├── neural_memory.py    # Updated to use Triton kernels
│   └── ...
└── opt/
    ├── deep_nested_optimizer.py  # Updated for branchless NaN handling
    └── ...
```

---

## Testing Strategy

1. **Numerical Verification**: Compare outputs to PyTorch reference
2. **Gradient Check**: Verify gradients match autograd.grad()
3. **Performance Benchmark**: Compare kernel time vs PyTorch
4. **Full Training Test**: Verify training dynamics unchanged

---

## References

- [Triton Language Documentation](https://triton-lang.org/)
- [Flash Attention Triton Implementation](https://github.com/Dao-AILab/flash-attention)
- [PyTorch Triton Tutorials](https://pytorch.org/tutorials/intermediate/triton.html)
