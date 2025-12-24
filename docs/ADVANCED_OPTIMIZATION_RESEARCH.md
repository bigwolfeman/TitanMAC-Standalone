# Advanced Optimization Research for TitanMAC + DeepNestedOptimizer

**Date**: December 20, 2025
**Status**: Research complete, implementation paths identified

This document catalogs advanced optimization techniques researched for improving TitanMAC training performance, particularly around the neural memory's `torch.autograd.grad()` constraint.

---

## Executive Summary

The neural memory architecture uses `torch.autograd.grad(..., retain_graph=True)` which blocks:
- CUDA graphs
- fullgraph=True in torch.compile
- donated_buffer optimization

Research identified several paths forward, from quick wins to custom CUDA kernels.

---

## 1. Quick Wins (Implemented)

### 1.1 torch.compile with dynamic=False
```python
model = torch.compile(model, mode='reduce-overhead', dynamic=False)
```
- Prevents recompilation on same-shape inputs
- Works with graph breaks at neural memory boundary

### 1.2 @torch._dynamo.disable on neural_memory.update()
```python
@torch._dynamo.disable
def update(self, x, theta_t=None, return_stats=False):
    grads = torch.autograd.grad(...)  # Non-traceable
```
- Allows rest of model to be compiled
- Creates graph break at memory update boundary

### 1.3 donated_buffer=False
```python
torch._functorch.config.donated_buffer = False
```
- Required for retain_graph=True compatibility

---

## 2. Compiled Autograd (To Try)

Enable compiled autograd for better backward pass optimization:

```python
# Global enable
torch._dynamo.config.compiled_autograd = True

# Or context manager for specific backward
with torch._dynamo.compiled_autograd.enable(torch.compile(fullgraph=True)):
    loss.backward()
```

**Expected Impact**: Better fusion of backward operations
**Risk**: May cause recompilations, doesn't solve forward-pass grad() issue

---

## 3. torch.func.grad Refactor (Medium Effort)

Rewrite neural memory using functional API for fullgraph compatibility:

```python
from torch.func import grad, functional_call

def make_functional_memory(memory_module):
    def forward(params, buffers, x):
        return functional_call(memory_module, (params, buffers), x)

    def memory_step(params, buffers, x, lr):
        grad_fn = grad(lambda p: forward(p, buffers, x).sum())
        g = grad_fn(params)
        new_params = {k: p - lr * g[k] for k, p in params.items()}
        return new_params

    return torch.compile(memory_step, fullgraph=True)
```

**Effort**: 2-4 days refactoring
**Impact**: Could enable fullgraph=True

---

## 4. PyLO CUDA Kernels (Reference Implementation)

The [PyLO library](https://github.com/Belilovsky-Lab/pylo) provides production CUDA kernels for learned optimizers. Key findings:

### 4.1 Kernel Architecture

**learned_optimizer.cu:**
- Block size: 256 threads
- Input: 39 features (28 optimizer stats + 11 temporal embeddings)
- MLP: 39 → 32 (ReLU) → 32 (ReLU) → 2 (direction, magnitude)

**velo_kernel.cu:**
- Block size: 256 threads
- Input: 30 features
- MLP: 30 → 4 (ReLU) → 4 (ReLU) → 3 (direction, magnitude, scale)

### 4.2 Key Design Patterns

**Two-Phase Execution:**
1. `lo_kernel()` - Accumulation phase: warp-level reductions for statistics
2. `lo_kernel_apply()` - Application phase: MLP inference + parameter update

**Feature Engineering:**
```cuda
__device__ void populate_vector_inp() {
    // 28 features from:
    // - Gradients and parameters
    // - Momentum buffer values (three decay states)
    // - Velocity with reciprocal sqrt normalization
    // - Row/column factorization terms
}

__device__ float tanh_embedding(float x, int idx) {
    // 11 timescales: 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000
    return tanh(x / timescales[idx]);
}
```

**Update Formula:**
```cuda
param -= lr * (output[0] * exp(output[1] * exp_mult) * step_mult);
```

### 4.3 Adaptation Path for Neural Memory

The PyLO pattern can be adapted for neural memory:

1. **Extend feature dimension**: Add memory state to feature vector
2. **Modify MLP input**: 39 → 39 + mem_features
3. **Train custom MetaMLP**: With memory-aware inputs
4. **Fuse update**: Memory gradient + parameter update in single kernel

**Key Insight**: PyLO's MLP weights are **frozen** (no gradient tracking needed). This simplifies CUDA kernel design significantly.

### 4.4 File Structure
```
/pylo/csrc/
├── learned_optimizer.cu    # 2-layer MLP kernel
└── velo_kernel.cu          # 3-layer MLP kernel (VeLO variant)
```

---

## 5. TorchOpt Library

[TorchOpt](https://github.com/metaopt/torchopt) provides differentiable optimization:

### 5.1 Implicit Gradient (Avoids Unrolling)
```python
@torchopt.implicit.custom_root(optimality_condition)
def solve_inner(params, data):
    return converged_params

# Gradients via implicit function theorem
outer_grad = torch.autograd.grad(solve_inner(...), outer_params)
```

**Best for**: If memory update can be framed as optimization problem
**Reported speedup**: 5-20x on GPU vs naive PyTorch

### 5.2 Functional Optimizer
```python
inner_opt = torchopt.adam(lr=0.01)
opt_state = inner_opt.init(model_params)

def inner_loop(params, opt_state, data):
    loss = compute_loss(params, data)
    grads = torchopt.extract_state_dict(model).grad
    updates, new_state = inner_opt.update(grads, opt_state)
    return torchopt.apply_updates(params, updates), new_state
```

---

## 6. Custom Triton Kernel Path

For maximum performance, write a Triton kernel:

```python
@torch.library.triton_op("titans::memory_update", mutates_args={})
def memory_update_triton(memory_state, input, grad_weights):
    # Triton kernel implementation
    ...

def backward(ctx, grad_output):
    return memory_update_backward_triton(ctx.saved_tensors, grad_output)

memory_update_triton.register_autograd(backward, setup_context=setup_context)
```

**Key Pattern**: Use `torch.library.triton_op` (not `custom_op`) for torch.compile visibility.

**Reference**: Flash Attention uses this pattern for fused attention + backward.

---

## 7. Priority Recommendations

| Priority | Approach | Effort | Impact | Status |
|----------|----------|--------|--------|--------|
| 1 | compiled_autograd | 1 line | Medium | Try now |
| 2 | Reduce eval overhead | Config | 20-30% | Ready |
| 3 | torch.func.grad refactor | 2-4 days | High | Future |
| 4 | PyLO-style CUDA kernel | 1-2 weeks | Highest | Future |
| 5 | Custom Triton kernel | 1-2 weeks | Highest | Future |

---

## 8. Why fullgraph=True Cannot Work

```
fullgraph=True requirement: NO graph breaks allowed
    ↓
torch.autograd.grad() is non-traceable by dynamo
    ↓
@torch._dynamo.disable creates a graph break
    ↓
Graph break + fullgraph=True = ERROR
    ↓
These are MUTUALLY EXCLUSIVE
```

**No workaround exists** within current PyTorch. Options are:
1. Accept graph breaks (current approach)
2. Refactor to torch.func.grad (functional API)
3. Custom CUDA/Triton kernel (bypass autograd entirely)

---

## 9. Custom Triton Kernel Design (Future Implementation)

Unlike PyLO which freezes the optimizer MLP (defeating the purpose of learning), our design requires **live weight updates** during training. Here's the kernel architecture:

### 9.1 Fused Forward + Backward + Update Kernel

```python
@triton.jit
def neural_memory_update_kernel(
    # Memory MLP weights (MUTABLE - key difference from PyLO)
    W1_ptr, B1_ptr,  # Layer 1: [d_in, hidden]
    W2_ptr, B2_ptr,  # Layer 2: [hidden, d_out]

    # Input/target for this update
    key_ptr,         # Query key [batch, d_in]
    value_ptr,       # Target value [batch, d_out]

    # Hyperparameters
    lr: tl.constexpr,
    d_in: tl.constexpr,
    hidden: tl.constexpr,
    d_out: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 64,
):
    """
    Fused kernel: Forward + Loss + Backward + Weight Update
    No Python autograd needed - gradients computed in kernel.
    """
    pid = tl.program_id(0)

    # ============ FORWARD PASS ============
    # Layer 1: hidden = ReLU(key @ W1 + B1)
    # Layer 2: output = hidden @ W2 + B2

    # ============ LOSS ============
    # MSE: d_loss/d_output = 2 * (output - target) / batch_size

    # ============ BACKWARD PASS ============
    # Manual backprop through 2-layer MLP
    # Gradient w.r.t. W2, B2: d_loss/d_W2 = hidden^T @ d_output
    # Gradient through ReLU: d_hidden = (W2 @ d_output) * (hidden_pre > 0)
    # Gradient w.r.t. W1, B1: d_loss/d_W1 = key^T @ d_hidden

    # ============ WEIGHT UPDATE ============
    # Atomic updates: W -= lr * grad
    tl.atomic_add(W1_ptr + offset, -lr * grad_w1)
    tl.atomic_add(W2_ptr + offset, -lr * grad_w2)
```

### 9.2 Why This Design

| Problem | PyTorch autograd | Our Triton kernel |
|---------|------------------|-------------------|
| retain_graph=True | Required | Not needed |
| CUDA graphs | Blocked | Compatible |
| fullgraph=True | Blocked | Compatible |
| Graph breaks | Yes | None |
| Kernel launches | Many | One fused |

### 9.3 Key Differences from PyLO

| PyLO (Broken Design) | Our Design |
|----------------------|------------|
| Frozen MLP weights | **Live updating weights** |
| Pre-trained update rule | **Learns during training** |
| No gradients for MLP | **Gradients computed in kernel** |
| Just inference | **Full forward + backward + update** |

### 9.4 Implementation Effort

- Basic 2-layer MLP kernel: 1-2 days
- With momentum/Adam state: 3-4 days
- Production quality + testing: 1 week

### 9.5 Usage Pattern

```python
# Before (Python autograd, slow, breaks compilation)
def update(self, x):
    loss = self.compute_loss(x)
    grads = torch.autograd.grad(loss, self.memory_mlp.parameters(),
                                 retain_graph=True)  # THE PROBLEM
    for p, g in zip(self.memory_mlp.parameters(), grads):
        p.data -= self.lr * g

# After (fused Triton kernel, fast, compilation friendly)
def update(self, key, value):
    neural_memory_update_kernel[grid](...)  # One kernel, no autograd
```

---

## 10. References

**PyTorch Documentation:**
- [Compiled Autograd Tutorial](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)
- [torch.func.grad](https://docs.pytorch.org/docs/stable/generated/torch.func.grad.html)
- [torch.library](https://docs.pytorch.org/docs/stable/library.html)

**Libraries:**
- [PyLO](https://github.com/Belilovsky-Lab/pylo) - Learned optimizers with CUDA kernels
- [TorchOpt](https://github.com/metaopt/torchopt) - Differentiable optimization
- [higher](https://github.com/facebookresearch/higher) - Higher-order gradients

**GitHub Issues:**
- [Dynamo should support torch.autograd.grad #167729](https://github.com/pytorch/pytorch/issues/167729)
