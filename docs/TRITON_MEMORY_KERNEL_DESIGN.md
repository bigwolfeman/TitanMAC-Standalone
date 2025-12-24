# Neural Memory Gradient Alternatives

**Date**: December 20, 2025
**Status**: Implementation Ready
**Goal**: Replace `torch.autograd.grad(..., retain_graph=True)` with fullgraph-compatible alternatives

---

## Executive Summary

**MAJOR FINDING**: `torch.func.grad` works with `torch.compile(fullgraph=True)`!

This discovery changes the recommendation. We now have THREE viable approaches:

| Approach | Complexity | fullgraph | Performance | Recommendation |
|----------|------------|-----------|-------------|----------------|
| `torch.func.grad` | Low | YES | Good | **RECOMMENDED** |
| Manual backward | Medium | YES | Good | Fallback |
| Triton kernels | High | YES | Best | Future optimization |

### Recommended Implementation Path

1. **Immediate**: Use `torch.func.grad` (implemented in `manual_backward.py`)
2. **If needed**: Fall back to manual backward for debugging
3. **Future**: Triton kernels for maximum performance

### Implementation Location

```
titans_core/memory/manual_backward.py
├── FunctionalGradMemoryUpdate    # RECOMMENDED - uses torch.func.grad
├── ManualBackwardMemoryUpdate    # Fallback - pure PyTorch manual backprop
└── verify_*()                    # Verification functions
```

---

## 1. Problem Statement

The neural memory's `update()` method uses:
```python
grads = torch.autograd.grad(
    loss,
    self.memory_mlp.parameters(),
    retain_graph=True,  # THE PROBLEM
    create_graph=False,
)
```

This **blocks**:
- `torch.compile(fullgraph=True)` - dynamo cannot trace `autograd.grad()`
- CUDA graphs - dynamic graph structure
- `donated_buffer` optimization - needs `retain_graph=True`

**Our solution**: Compute gradients manually in a fused Triton kernel, bypassing autograd entirely.

---

## 2. Mathematical Formulation

### 2.1 Forward Pass (MLP with 2 layers)

Given:
- **Keys**: `k ∈ ℝ^(N × D)` where N = batch×seq, D = d_memory
- **Values**: `v ∈ ℝ^(N × D)` (target)
- **Weights**: `W1 ∈ ℝ^(H × D)`, `B1 ∈ ℝ^H`, `W2 ∈ ℝ^(D × H)`, `B2 ∈ ℝ^D`

Forward:
```
z1 = k @ W1.T + B1           # [N, H] pre-activation
h = silu(z1)                  # [N, H] post-activation (hidden)
y = h @ W2.T + B2            # [N, D] output
L = (1/ND) * ||y - v||²      # scalar loss (MSE)
```

### 2.2 Backward Pass (Manual Gradients)

Working backwards from loss:

**Step 1: Output gradient**
```
∂L/∂y = (2/ND) * (y - v)     # [N, D]
```

**Step 2: Layer 2 gradients**
```
∂L/∂W2 = (∂L/∂y).T @ h       # [D, H]  (matmul reduction over N)
∂L/∂B2 = sum(∂L/∂y, dim=0)   # [D]     (reduction over N)
∂L/∂h = (∂L/∂y) @ W2         # [N, H]  (standard matmul)
```

**Step 3: SiLU backward**

SiLU: `silu(x) = x * σ(x)` where `σ(x) = 1/(1 + exp(-x))`

Derivative: `silu'(x) = σ(x) * (1 + x * (1 - σ(x)))`

```
σ1 = sigmoid(z1)                          # [N, H]
silu_grad = σ1 * (1 + z1 * (1 - σ1))      # [N, H]
∂L/∂z1 = (∂L/∂h) * silu_grad              # [N, H] element-wise
```

**Step 4: Layer 1 gradients**
```
∂L/∂W1 = (∂L/∂z1).T @ k      # [H, D]  (matmul reduction over N)
∂L/∂B1 = sum(∂L/∂z1, dim=0)  # [H]     (reduction over N)
```

### 2.3 Gradient Clipping
```
g = flatten([∂L/∂W1, ∂L/∂B1, ∂L/∂W2, ∂L/∂B2])
norm = ||g||₂
if norm > max_norm:
    g = g * (max_norm / norm)
```

### 2.4 Momentum Update (Eq. 13 from Titans paper)
```
S = η * S - θ * g
```

### 2.5 Weight Update (Eq. 14 from Titans paper)
```
W_flat = (1 - α) * W_flat + S
```

---

## 3. Kernel Architecture

### 3.1 Why Not Single Monolithic Kernel?

The backward pass involves:
1. **Two large matmul reductions** (∂W1, ∂W2) over batch dimension
2. **Element-wise operations** (SiLU backward, output gradient)
3. **Global reduction** (gradient norm)
4. **Element-wise updates** (momentum, weights)

A single kernel would have:
- Poor occupancy (serial dependencies)
- Complex synchronization
- Register pressure issues

### 3.2 Proposed Multi-Kernel Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Orchestration                      │
│                                                              │
│  1. Forward pass (PyTorch) → save z1, h, y                  │
│  2. Call Triton kernels in sequence                         │
│  3. Gates computed in PyTorch (can be parallelized)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Kernel 1: output_grad_and_layer2_back            │
│                                                              │
│  Inputs: y, v, h, W2                                        │
│  Outputs: d_y, d_h, d_W2, d_B2                              │
│                                                              │
│  Operations:                                                 │
│  - d_y = (2/ND) * (y - v)         [element-wise]            │
│  - d_h = d_y @ W2                  [matmul]                 │
│  - d_W2 = d_y.T @ h               [matmul reduction]        │
│  - d_B2 = sum(d_y)                [reduction]               │
│                                                              │
│  Tiling: BLOCK_N=128, BLOCK_D=64, BLOCK_H=64               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Kernel 2: silu_back_and_layer1_back              │
│                                                              │
│  Inputs: d_h, z1, k                                         │
│  Outputs: d_z1, d_W1, d_B1                                  │
│                                                              │
│  Operations:                                                 │
│  - σ = sigmoid(z1)                [element-wise]            │
│  - silu_grad = σ * (1 + z1*(1-σ)) [element-wise]            │
│  - d_z1 = d_h * silu_grad         [element-wise]            │
│  - d_W1 = d_z1.T @ k              [matmul reduction]        │
│  - d_B1 = sum(d_z1)               [reduction]               │
│                                                              │
│  Tiling: BLOCK_N=128, BLOCK_H=64, BLOCK_D=64               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Kernel 3: grad_clip_momentum_update              │
│                                                              │
│  Inputs: d_W1, d_B1, d_W2, d_B2, W1, B1, W2, B2,           │
│          S_W1, S_B1, S_W2, S_B2, α, η, θ, max_norm         │
│                                                              │
│  Outputs: Updated W1, B1, W2, B2, S_W1, S_B1, S_W2, S_B2   │
│                                                              │
│  Operations:                                                 │
│  Phase A (parallel blocks, atomic add to partial_norm):     │
│  - Compute local ||g||² and atomic_add to global            │
│  - __syncthreads() / grid sync                              │
│                                                              │
│  Phase B (after sync):                                       │
│  - norm = sqrt(sum_partial_norms)                           │
│  - scale = min(1.0, max_norm / norm)                        │
│  - For each weight w, gradient g, momentum s:               │
│      s = η * s - θ * (g * scale)                            │
│      w = (1 - α) * w + s                                    │
│                                                              │
│  Tiling: BLOCK=1024 (1D), each block handles subset        │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Alternative: Fused Kernel 1+2

Since kernel 1 and 2 share data (d_h flows from 1→2), we can fuse them:

```
┌─────────────────────────────────────────────────────────────┐
│              Kernel A: full_backward_pass                    │
│                                                              │
│  Inputs: y, v, h, z1, k, W2                                 │
│  Outputs: d_W1, d_B1, d_W2, d_B2                            │
│                                                              │
│  Single kernel computes all gradients                        │
│  Keeps d_h, d_z1 in registers/shared memory                 │
│                                                              │
│  More complex but fewer kernel launches                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Kernel Implementations

### 4.1 Kernel 1: Output Grad and Layer 2 Backward

```python
@triton.jit
def kernel_output_grad_layer2_back(
    # Inputs (read-only)
    y_ptr,      # [N, D] output
    v_ptr,      # [N, D] target
    h_ptr,      # [N, H] hidden activation
    W2_ptr,     # [D, H] layer 2 weight

    # Outputs
    d_y_ptr,    # [N, D] output gradient (write)
    d_h_ptr,    # [N, H] hidden gradient (write)
    d_W2_ptr,   # [D, H] weight gradient (write, atomic add)
    d_B2_ptr,   # [D] bias gradient (write, atomic add)

    # Dimensions
    N, D, H,

    # Strides
    stride_y_n, stride_y_d,
    stride_h_n, stride_h_h,
    stride_W2_d, stride_W2_h,

    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Computes:
    1. d_y = (2/ND) * (y - v)
    2. d_h = d_y @ W2
    3. d_W2 += d_y.T @ h (atomic accumulation)
    4. d_B2 += sum(d_y, dim=0) (atomic accumulation)
    """
    # Grid: (N // BLOCK_N, D // BLOCK_D)
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Offsets for this tile
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Masks for boundary checking
    mask_n = offs_n < N
    mask_d = offs_d < D
    mask_nd = mask_n[:, None] & mask_d[None, :]

    # ==================== STEP 1: d_y = (2/ND) * (y - v) ====================
    # Load tiles
    y_ptrs = y_ptr + offs_n[:, None] * stride_y_n + offs_d[None, :] * stride_y_d
    v_ptrs = v_ptr + offs_n[:, None] * stride_y_n + offs_d[None, :] * stride_y_d

    y_tile = tl.load(y_ptrs, mask=mask_nd, other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_nd, other=0.0)

    scale = 2.0 / (N * D)
    d_y_tile = scale * (y_tile - v_tile)  # [BLOCK_N, BLOCK_D]

    # Store d_y
    d_y_ptrs = d_y_ptr + offs_n[:, None] * stride_y_n + offs_d[None, :] * stride_y_d
    tl.store(d_y_ptrs, d_y_tile, mask=mask_nd)

    # ==================== STEP 2: d_h = d_y @ W2 ====================
    # This is a matmul: [BLOCK_N, BLOCK_D] @ [D, H] -> [BLOCK_N, H]
    # We accumulate over D dimension

    offs_h = tl.arange(0, BLOCK_H)
    d_h_accum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        offs_d_inner = d_start + tl.arange(0, BLOCK_D)
        mask_d_inner = offs_d_inner < D

        # Load d_y tile for this D chunk
        d_y_ptrs_inner = d_y_ptr + offs_n[:, None] * stride_y_n + offs_d_inner[None, :] * stride_y_d
        d_y_inner = tl.load(d_y_ptrs_inner, mask=mask_n[:, None] & mask_d_inner[None, :], other=0.0)

        # Load W2 tile: [BLOCK_D, BLOCK_H]
        for h_start in range(0, H, BLOCK_H):
            offs_h_inner = h_start + tl.arange(0, BLOCK_H)
            mask_h_inner = offs_h_inner < H

            W2_ptrs = W2_ptr + offs_d_inner[:, None] * stride_W2_d + offs_h_inner[None, :] * stride_W2_h
            W2_tile = tl.load(W2_ptrs, mask=mask_d_inner[:, None] & mask_h_inner[None, :], other=0.0)

            # Accumulate: d_h[:, h_start:h_start+BLOCK_H] += d_y_inner @ W2_tile
            d_h_partial = tl.dot(d_y_inner, W2_tile)  # [BLOCK_N, BLOCK_H]

            # Store to appropriate position
            d_h_ptrs = d_h_ptr + offs_n[:, None] * stride_h_n + offs_h_inner[None, :] * stride_h_h
            if d_start == 0:
                tl.store(d_h_ptrs, d_h_partial, mask=mask_n[:, None] & mask_h_inner[None, :])
            else:
                # Atomic add for accumulation across D blocks
                tl.atomic_add(d_h_ptrs, d_h_partial, mask=mask_n[:, None] & mask_h_inner[None, :])

    # ==================== STEP 3: d_W2 += d_y.T @ h ====================
    # [D, N] @ [N, H] -> [D, H]
    # Each block contributes its local [BLOCK_D, BLOCK_H] chunk

    for h_start in range(0, H, BLOCK_H):
        offs_h_inner = h_start + tl.arange(0, BLOCK_H)
        mask_h_inner = offs_h_inner < H

        # Load h tile: [BLOCK_N, BLOCK_H]
        h_ptrs = h_ptr + offs_n[:, None] * stride_h_n + offs_h_inner[None, :] * stride_h_h
        h_tile = tl.load(h_ptrs, mask=mask_n[:, None] & mask_h_inner[None, :], other=0.0)

        # Compute d_y.T @ h for this tile
        # d_y_tile: [BLOCK_N, BLOCK_D], h_tile: [BLOCK_N, BLOCK_H]
        # d_y.T: [BLOCK_D, BLOCK_N] @ [BLOCK_N, BLOCK_H] = [BLOCK_D, BLOCK_H]
        d_W2_local = tl.dot(tl.trans(d_y_tile), h_tile)  # [BLOCK_D, BLOCK_H]

        # Atomic add to global d_W2
        d_W2_ptrs = d_W2_ptr + offs_d[:, None] * stride_W2_d + offs_h_inner[None, :] * stride_W2_h
        tl.atomic_add(d_W2_ptrs, d_W2_local, mask=mask_d[:, None] & mask_h_inner[None, :])

    # ==================== STEP 4: d_B2 += sum(d_y, dim=0) ====================
    # Reduce over N dimension
    d_B2_local = tl.sum(d_y_tile, axis=0)  # [BLOCK_D]

    # Atomic add to global d_B2
    d_B2_ptrs = d_B2_ptr + offs_d
    tl.atomic_add(d_B2_ptrs, d_B2_local, mask=mask_d)
```

### 4.2 Kernel 2: SiLU Backward and Layer 1 Backward

```python
@triton.jit
def kernel_silu_layer1_back(
    # Inputs (read-only)
    d_h_ptr,    # [N, H] hidden gradient
    z1_ptr,     # [N, H] pre-activation
    k_ptr,      # [N, D] keys (input)

    # Outputs
    d_W1_ptr,   # [H, D] weight gradient (atomic add)
    d_B1_ptr,   # [H] bias gradient (atomic add)

    # Dimensions
    N, H, D,

    # Strides
    stride_dh_n, stride_dh_h,
    stride_z1_n, stride_z1_h,
    stride_k_n, stride_k_d,
    stride_W1_h, stride_W1_d,

    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computes:
    1. σ = sigmoid(z1)
    2. silu_grad = σ * (1 + z1 * (1 - σ))
    3. d_z1 = d_h * silu_grad
    4. d_W1 += d_z1.T @ k (atomic accumulation)
    5. d_B1 += sum(d_z1, dim=0) (atomic accumulation)
    """
    # Grid: (N // BLOCK_N, H // BLOCK_H)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_n = offs_n < N
    mask_h = offs_h < H
    mask_nh = mask_n[:, None] & mask_h[None, :]

    # ==================== STEP 1-3: SiLU backward ====================
    # Load d_h and z1
    d_h_ptrs = d_h_ptr + offs_n[:, None] * stride_dh_n + offs_h[None, :] * stride_dh_h
    z1_ptrs = z1_ptr + offs_n[:, None] * stride_z1_n + offs_h[None, :] * stride_z1_h

    d_h_tile = tl.load(d_h_ptrs, mask=mask_nh, other=0.0)
    z1_tile = tl.load(z1_ptrs, mask=mask_nh, other=0.0)

    # σ = sigmoid(z1)
    sigma = tl.sigmoid(z1_tile)

    # silu_grad = σ * (1 + z1 * (1 - σ))
    silu_grad = sigma * (1.0 + z1_tile * (1.0 - sigma))

    # d_z1 = d_h * silu_grad
    d_z1_tile = d_h_tile * silu_grad  # [BLOCK_N, BLOCK_H]

    # ==================== STEP 4: d_W1 += d_z1.T @ k ====================
    # [H, N] @ [N, D] -> [H, D]

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load k tile: [BLOCK_N, BLOCK_D]
        k_ptrs = k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # d_z1.T @ k: [BLOCK_H, BLOCK_N] @ [BLOCK_N, BLOCK_D] = [BLOCK_H, BLOCK_D]
        d_W1_local = tl.dot(tl.trans(d_z1_tile), k_tile)

        # Atomic add to global d_W1
        d_W1_ptrs = d_W1_ptr + offs_h[:, None] * stride_W1_h + offs_d[None, :] * stride_W1_d
        tl.atomic_add(d_W1_ptrs, d_W1_local, mask=mask_h[:, None] & mask_d[None, :])

    # ==================== STEP 5: d_B1 += sum(d_z1, dim=0) ====================
    d_B1_local = tl.sum(d_z1_tile, axis=0)  # [BLOCK_H]

    d_B1_ptrs = d_B1_ptr + offs_h
    tl.atomic_add(d_B1_ptrs, d_B1_local, mask=mask_h)
```

### 4.3 Kernel 3: Gradient Clipping + Momentum + Weight Update

```python
@triton.jit
def kernel_grad_clip_momentum_update(
    # Gradients (read, will be scaled in-place conceptually)
    d_W1_ptr, d_B1_ptr, d_W2_ptr, d_B2_ptr,

    # Weights (read-write)
    W1_ptr, B1_ptr, W2_ptr, B2_ptr,

    # Momentum (read-write)
    S_W1_ptr, S_B1_ptr, S_W2_ptr, S_B2_ptr,

    # Scalars
    alpha_t,      # forget gate
    eta_t,        # decay gate
    theta,        # learning rate
    max_grad_norm,
    grad_norm_ptr,  # Input: pre-computed gradient norm

    # Sizes
    W1_size,  # H * D
    B1_size,  # H
    W2_size,  # D * H
    B2_size,  # D

    # Block size
    BLOCK: tl.constexpr,
):
    """
    For each parameter p with gradient g and momentum s:
    1. scale = min(1.0, max_grad_norm / grad_norm)
    2. g_clipped = g * scale
    3. s = η * s - θ * g_clipped
    4. p = (1 - α) * p + s
    """
    pid = tl.program_id(0)

    # Load grad norm
    grad_norm = tl.load(grad_norm_ptr)
    scale = tl.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))

    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Determine which parameter we're updating based on offset
    total_params = W1_size + B1_size + W2_size + B2_size

    mask = offs < total_params

    # This is simplified - actual implementation would need to handle
    # different memory layouts for each parameter tensor
    # ... (implementation details for routing to correct buffers)

    # For now, assume flat memory layout:
    # [W1_flat | B1 | W2_flat | B2]

    # Load gradient, momentum, weight
    # Apply: s = η * s - θ * g * scale
    #        w = (1 - α) * w + s
    # Store updated s, w
```

---

## 5. Gradient Norm Computation

Before Kernel 3, we need the gradient norm. Options:

### Option A: Separate Reduction Kernel
```python
@triton.jit
def kernel_compute_grad_norm_squared(
    d_W1_ptr, d_B1_ptr, d_W2_ptr, d_B2_ptr,
    partial_sums_ptr,  # Output: partial ||g||² from each block
    W1_size, B1_size, W2_size, B2_size,
    BLOCK: tl.constexpr,
):
    """Each block computes local sum of squares, writes to partial_sums."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Load elements from all gradient tensors (based on offset)
    # Compute local_sum = sum(g[offs]^2)
    # Store to partial_sums[pid]
```

Then reduce partial sums on CPU or with another kernel:
```python
grad_norm = torch.sqrt(partial_sums.sum())
```

### Option B: Fused with Atomic (Simpler but More Contention)
```python
# In Kernel 3:
local_norm_sq = tl.sum(g_tile * g_tile)
tl.atomic_add(global_norm_sq_ptr, local_norm_sq)
# Grid sync
# Then compute scale
```

---

## 6. Python Wrapper

```python
import triton
import torch
import torch.nn.functional as F

class TritonMemoryUpdate:
    """
    Drop-in replacement for torch.autograd.grad() in neural memory.

    Computes MLP gradients manually via Triton, enabling fullgraph=True.
    """

    def __init__(self, d_memory: int, hidden_dim: int):
        self.d_memory = d_memory
        self.hidden_dim = hidden_dim

        # Pre-allocate gradient buffers
        self.d_W1 = None
        self.d_B1 = None
        self.d_W2 = None
        self.d_B2 = None

    def __call__(
        self,
        k: torch.Tensor,          # [N, D] keys
        v: torch.Tensor,          # [N, D] values (target)
        z1: torch.Tensor,         # [N, H] pre-activation (saved from forward)
        h: torch.Tensor,          # [N, H] hidden (saved from forward)
        y: torch.Tensor,          # [N, D] output (saved from forward)
        W1: torch.Tensor,         # [H, D] layer 1 weight
        B1: torch.Tensor,         # [H] layer 1 bias
        W2: torch.Tensor,         # [D, H] layer 2 weight
        B2: torch.Tensor,         # [D] layer 2 bias
        momentum_S: torch.Tensor, # [n_params] momentum buffer
        alpha_t: float,           # forget gate
        eta_t: float,             # decay gate
        theta: float,             # learning rate
        max_grad_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute gradients and update weights in-place.

        Returns:
            grad_norm: The gradient norm before clipping
        """
        N, D = k.shape
        H = self.hidden_dim

        # Allocate gradient buffers if needed
        if self.d_W1 is None or self.d_W1.shape != W1.shape:
            self.d_W1 = torch.zeros_like(W1)
            self.d_B1 = torch.zeros_like(B1)
            self.d_W2 = torch.zeros_like(W2)
            self.d_B2 = torch.zeros_like(B2)
            self.d_y = torch.zeros_like(y)
            self.d_h = torch.zeros_like(h)
        else:
            # Zero gradients
            self.d_W1.zero_()
            self.d_B1.zero_()
            self.d_W2.zero_()
            self.d_B2.zero_()

        # Block sizes (tuned for typical dimensions)
        BLOCK_N = 128
        BLOCK_D = 64
        BLOCK_H = 64

        # Grid dimensions
        grid_k1 = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
        grid_k2 = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

        # Kernel 1: d_y, d_h, d_W2, d_B2
        kernel_output_grad_layer2_back[grid_k1](
            y, v, h, W2,
            self.d_y, self.d_h, self.d_W2, self.d_B2,
            N, D, H,
            y.stride(0), y.stride(1),
            h.stride(0), h.stride(1),
            W2.stride(0), W2.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H,
        )

        # Kernel 2: d_W1, d_B1
        kernel_silu_layer1_back[grid_k2](
            self.d_h, z1, k,
            self.d_W1, self.d_B1,
            N, H, D,
            self.d_h.stride(0), self.d_h.stride(1),
            z1.stride(0), z1.stride(1),
            k.stride(0), k.stride(1),
            W1.stride(0), W1.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
        )

        # Compute gradient norm
        grad_norm_sq = (
            self.d_W1.pow(2).sum() +
            self.d_B1.pow(2).sum() +
            self.d_W2.pow(2).sum() +
            self.d_B2.pow(2).sum()
        )
        grad_norm = grad_norm_sq.sqrt()

        # Clip and apply updates
        scale = torch.clamp(max_grad_norm / (grad_norm + 1e-8), max=1.0)

        # Flatten gradients
        flat_grad = torch.cat([
            (self.d_W1 * scale).view(-1),
            (self.d_B1 * scale).view(-1),
            (self.d_W2 * scale).view(-1),
            (self.d_B2 * scale).view(-1),
        ])

        # Momentum update: S = η * S - θ * g
        momentum_S.mul_(eta_t)
        momentum_S.add_(flat_grad, alpha=-theta)

        # Weight update: W = (1 - α) * W + S
        # Unflatten momentum to update each weight
        offset = 0
        for p in [W1, B1, W2, B2]:
            numel = p.numel()
            p.mul_(1.0 - alpha_t)
            p.add_(momentum_S[offset:offset + numel].view(p.shape))
            offset += numel

        return grad_norm
```

---

## 7. Integration with NeuralMemory

```python
# In neural_memory.py

class NeuralMemory(nn.Module):
    def __init__(self, ...):
        ...
        # Initialize Triton updater
        self.triton_update = TritonMemoryUpdate(self.d_memory, self.d_memory)

    # REMOVE @torch._dynamo.disable - no longer needed!
    def update(self, x, theta_t=None, return_stats=False):
        """
        Now compatible with torch.compile(fullgraph=True)!
        """
        theta = theta_t if theta_t is not None else self.theta

        # Project to key/value space
        k = self.W_K(x)  # [batch, seq, d_memory]
        v = self.W_V(x)  # [batch, seq, d_memory]

        # Reshape for processing
        N = k.shape[0] * k.shape[1]
        k_flat = k.view(N, -1)
        v_flat = v.view(N, -1)

        # Get MLP weights
        W1, B1 = self.memory_mlp.mlp[0].weight, self.memory_mlp.mlp[0].bias
        W2, B2 = self.memory_mlp.mlp[2].weight, self.memory_mlp.mlp[2].bias

        # Forward pass (save intermediates)
        z1 = F.linear(k_flat, W1, B1)  # Pre-activation
        h = F.silu(z1)                  # Hidden
        y = F.linear(h, W2, B2)         # Output
        loss = F.mse_loss(y, v_flat)

        # Compute gates
        x_pooled = x.mean(dim=1)
        alpha_t = self.forget_gate(x_pooled).mean()
        eta_t = self.decay_gate(x_pooled).mean()

        # Triton backward + update (replaces autograd.grad)
        with torch.no_grad():
            grad_norm = self.triton_update(
                k_flat, v_flat, z1, h, y,
                W1, B1, W2, B2,
                self.momentum_S,
                alpha_t.item(), eta_t.item(), theta,
            )

        if return_stats:
            return {
                "loss": loss,
                "alpha_t": alpha_t.item(),
                "eta_t": eta_t.item(),
                "grad_norm": grad_norm.item(),
                "grad_clipped": grad_norm > 1.0,
            }
        return loss
```

---

## 8. Expected Performance Gains

| Metric | Before (autograd) | After (Triton) |
|--------|-------------------|----------------|
| Kernel launches | ~20+ (autograd graph) | 2-3 |
| retain_graph overhead | 2x peak memory | None |
| CUDA graph compatible | No | Yes |
| fullgraph=True | No | Yes |
| donated_buffer | No | Yes |

**Estimated speedup**: 2-5x for memory update operation (which runs every step).

---

## 9. Implementation Phases

### Phase 1: Reference Implementation (Python)
1. Implement manual backward in pure PyTorch (no autograd.grad)
2. Verify numerical equivalence with autograd
3. Benchmark vs autograd version

### Phase 2: Triton Kernels
1. Implement Kernel 1 (output_grad_layer2_back)
2. Implement Kernel 2 (silu_layer1_back)
3. Implement gradient norm reduction
4. Test each kernel in isolation

### Phase 3: Fusion + Integration
1. Fuse Kernel 1+2 if beneficial
2. Add gradient clipping + momentum + update kernel
3. Integrate with NeuralMemory class
4. Remove @torch._dynamo.disable decorator
5. Test with torch.compile(fullgraph=True)

### Phase 4: Optimization
1. Tune block sizes for target GPU
2. Profile with ncu/nsight
3. Optimize memory access patterns
4. Consider using tensor cores for matmuls

---

## 10. Key Differences from PyLO

| Aspect | PyLO | Our Design |
|--------|------|------------|
| MLP weights | **Frozen** (pre-trained) | **Live updating** |
| Gradients | None (inference only) | Computed in kernel |
| Purpose | Apply learned update rule | Learn update rule online |
| Backward pass | Not needed | Full backprop |
| torch.compile | Not applicable | fullgraph=True enabled |

PyLO's frozen weights make their kernel simpler (just forward inference), but defeats the purpose of a learned optimizer. Our design is more complex but preserves the learning capability.

---

## 11. Numerical Verification

Before deployment, verify numerical equivalence:

```python
def verify_triton_vs_autograd(memory_module, x):
    """Verify Triton backward matches autograd."""

    # Method 1: autograd.grad
    loss1 = memory_module.compute_loss(x)
    grads_autograd = torch.autograd.grad(
        loss1, memory_module.memory_mlp.parameters(),
        retain_graph=True,
    )

    # Method 2: Triton manual backward
    k = memory_module.W_K(x).view(-1, memory_module.d_memory)
    v = memory_module.W_V(x).view(-1, memory_module.d_memory)
    W1, B1 = memory_module.memory_mlp.mlp[0].weight, memory_module.memory_mlp.mlp[0].bias
    W2, B2 = memory_module.memory_mlp.mlp[2].weight, memory_module.memory_mlp.mlp[2].bias

    z1 = F.linear(k, W1, B1)
    h = F.silu(z1)
    y = F.linear(h, W2, B2)

    # Run Triton kernels (without update)
    d_W1, d_B1, d_W2, d_B2 = triton_backward(k, v, z1, h, y, W2)
    grads_triton = [d_W1, d_B1, d_W2, d_B2]

    # Compare
    for g_auto, g_tri in zip(grads_autograd, grads_triton):
        assert torch.allclose(g_auto, g_tri, rtol=1e-4, atol=1e-6), \
            f"Mismatch: {(g_auto - g_tri).abs().max()}"

    print("Numerical verification passed!")
```

---

## 12. References

- [Triton Documentation](https://triton-lang.org/main/index.html)
- [Titans Paper](https://arxiv.org/abs/2501.00663) - Neural memory equations
- [PyTorch torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- [Flash Attention Triton](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
