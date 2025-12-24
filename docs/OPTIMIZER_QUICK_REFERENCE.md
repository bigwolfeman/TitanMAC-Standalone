# DeepNestedOptimizer Quick Reference

## Constructor Parameters

```python
from titans_core.opt import DeepNestedOptimizer

optimizer = DeepNestedOptimizer(
    model=model,                    # Required: nn.Module to optimize

    # === Learning Rates ===
    base_lr=3e-4,                   # Base learning rate for model params
    meta_lr=1e-4,                   # Learning rate for meta-learner updates

    # === Architecture ===
    momentum_hidden_dim=64,         # Hidden dimension of MomentumMLP
    momentum_num_layers=4,          # Number of layers in MomentumMLP (TitanMAC winner: 4)
    controller_hidden_dim=32,       # Hidden dimension of NestedController
    controller_num_layers=5,        # Number of layers in controller (TitanMAC winner: 5)

    # === Modes ===
    mode='simple',                  # 'simple' | 'explicit' | 'implicit'
    use_cms_updates=False,          # Enable CMS (surrogate loss) updates
    use_preprocessing=True,         # Enable DirectUpdateMLP preprocessing

    # === Regularization ===
    weight_decay=0.01,              # Weight decay for base optimizer
    max_grad_norm=1.0,              # Gradient clipping threshold (0 = disabled)

    # === Meta Updates ===
    meta_update_freq=50,            # Steps between meta-learner updates
    k_unroll=5,                     # Unrolling steps for meta-gradient

    # === Performance (NEW) ===
    use_compile=False,              # Apply torch.compile to controller/MLP
    controller_update_freq=1,       # Steps between controller updates (1 = every step)
    use_cuda_graph=False,           # Enable CUDA Graph capture (42% speedup)
    cuda_graph_warmup_steps=3,      # Warmup steps before graph capture
)
```

---

## Winning Configurations

### MoE + DeepNestedOptimizer

```python
# Best from grid search (M=3, C=2)
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    meta_lr=1e-4,
    momentum_num_layers=3,      # Shallower works better for MoE
    controller_num_layers=2,
    mode='explicit',
)
```

### TitanMAC + DeepNestedOptimizer

```python
# Best from grid search (M=4, C=5)
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    meta_lr=1e-4,
    momentum_num_layers=4,      # Deeper works better for TitanMAC
    controller_num_layers=5,
    mode='explicit',
)
```

---

## Performance Tuning

### Maximum Speed

```python
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    use_compile=True,               # Fuse kernels
    use_cuda_graph=True,            # 42% speedup from graph replay
    controller_update_freq=10,      # Batch controller updates
)
```

### Minimum Memory

```python
optimizer = DeepNestedOptimizer(
    model=model,
    base_lr=3e-4,
    use_cms_updates=False,          # No CMS momentum buffers
    momentum_num_layers=2,          # Smaller networks
    controller_num_layers=2,
)
```

---

## Key Methods

```python
# Standard training loop
optimizer.zero_grad()
loss = model(x, y)['loss']
loss.backward()
optimizer.step(loss.item())         # Pass loss value for meta-learning

# Get current LR multipliers
multipliers = optimizer.get_lr_multipliers()  # Tensor[n_groups]

# Get momentum statistics
stats = optimizer.get_momentum_stats()
# {'momentum_total_norm': float, 'param_group_norms': list}

# Manual meta update
optimizer._update_meta_components(loss_value)
```

---

## Memory Usage

| Component | Memory (168M param model) |
|-----------|---------------------------|
| Model params (fp32) | 672 MB |
| Gradients | 672 MB |
| AdamW state (m, v) | 1.34 GB |
| CMS momentum (3 levels) | 2.02 GB |
| Controller + MomentumMLP | ~2 MB |
| **Total (CMS enabled)** | ~4.7 GB |
| **Total (CMS disabled)** | ~2.7 GB |

---

## Common Issues

### 1. In-Place Operation Crash (CRITICAL)

**Error:**
```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [N]] is at version X
```

**Cause:** Meta-learning backward tries to go through model parameters that were modified by `optimizer.step()`.

**Fix:** Ensure these functions detach model params from the graph:
- `_compute_group_stats()`: Wrap in `torch.no_grad()`, return `stats.detach()`
- `_compute_mlp_proxy_loss()`: Detach `grad`, `prev_momentum`, `ema_target`

See `PERFORMANCE_OPTIMIZATIONS.md` Section 6 for full details.

### 2. High VRAM Usage

```python
# Disable CMS if memory-constrained
optimizer = DeepNestedOptimizer(..., use_cms_updates=False)
```

### 3. Slow Training

```python
# Enable performance optimizations
optimizer = DeepNestedOptimizer(
    ...,
    use_compile=True,
    controller_update_freq=10,
)
```

### 4. Unstable Training

```python
# Use shallower networks, lower meta_lr
optimizer = DeepNestedOptimizer(
    ...,
    momentum_num_layers=2,
    controller_num_layers=2,
    meta_lr=5e-5,
)
```

### 5. Loss Stuck / Not Decreasing

```python
# Try explicit mode, increase base_lr
optimizer = DeepNestedOptimizer(
    ...,
    mode='explicit',
    base_lr=1e-3,
)
```

### 6. CUDA Graph Memory Corruption

**Symptom:** With `use_cuda_graph=True`, loss randomly drops to near-zero (0.001-0.02) then recovers.

**Cause:** Gradient clipping was inside the CUDA graph, but gradient tensor addresses change between backward passes.

**Fix:** Gradient clipping must be **outside** the graph. The graph should only contain `optimizer.step()`.

See `PERFORMANCE_OPTIMIZATIONS.md` Section 7 for details.

### 7. torch.compile + Neural Memory Crash

**Symptom:** With `torch.compile()` enabled, training crashes at meta-update step with:
```
RuntimeError: This backward function was compiled with non-empty donated buffers
which requires create_graph=False and retain_graph=False.
```

**Cause:** torch.compile's `donated_buffer` optimization conflicts with neural memory's `torch.autograd.grad(..., retain_graph=True)` call.

**Fix:**
1. Decorate `neural_memory.update()` with `@torch._dynamo.disable`
2. Disable donated buffers before compiling
3. Do NOT use `fullgraph=True` (creates graph break conflict)

```python
# In neural_memory.py:
@torch._dynamo.disable
def update(self, x, theta_t=None, return_stats=False):
    ...  # Contains torch.autograd.grad()

# In training script:
import torch._functorch.config
torch._functorch.config.donated_buffer = False
# Note: fullgraph=True is NOT compatible with @dynamo.disable
model = torch.compile(model, mode='reduce-overhead', dynamic=False)
```

---

## Comparison vs Muon+AdamW

| Metric | Muon+AdamW | DeepNested | Notes |
|--------|------------|------------|-------|
| Val Loss (4800 steps) | **2.73** | 3.50 | Muon still wins |
| VRAM | 10.9 GB | 20.2 GB | 2x overhead |
| Step time | ~180ms | ~280ms | 1.5x slower |
| Convergence | Fast | Slower | Needs more steps |

**Current recommendation**: Use Muon+AdamW for production. DeepNested is experimental.

---

## Files

| File | Description |
|------|-------------|
| `titans_core/opt/deep_nested_optimizer.py` | Main optimizer class |
| `titans_core/opt/nested_controller.py` | LR multiplier network |
| `titans_core/opt/momentum_mlp.py` | Per-element update network |
| `docs/GPU_PROFILING_REPORT.md` | Profiling analysis |
| `docs/PERFORMANCE_OPTIMIZATIONS.md` | Optimization details |
