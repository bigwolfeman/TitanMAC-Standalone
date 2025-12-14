# Nested Learning Implementation - Port Guide

**Date**: 2025-12-14
**Source**: `/mnt/BigAssDrive/00projects/00DeepNet/111Fin-TitanMAC/`
**Target**: `/mnt/BigAssDrive/00projects/00DeepNet/111TitanMAC-Standalone/`
**Reference**: NeurIPS 2025 - "Nested Learning: The Illusion of Deep Learning Architectures"

---

## Overview

This document describes the changes needed to implement the complete Nested Learning optimizer in the standalone TitanMAC repository.

### What Was Implemented

1. **DeepNestedOptimizer** - Unified optimizer combining:
   - L2RegressionMomentum (DMGD with learned momentum)
   - NestedController (learned LR multipliers)
   - ContinuumMemoryState (multi-frequency CMS updates)
   - Meta-learning via SimplifiedMetaTrainer and UnrolledMetaTrainer

2. **BCTrainer Integration** - Support for `optimizer_type: nested` in config

3. **Config Schema** - YAML configuration for all nested optimizer parameters

---

## Files to Copy/Create

### 1. NEW FILE: `titans_core/opt/deep_nested_optimizer.py`

**Location**: `titans_core/opt/deep_nested_optimizer.py`
**Size**: ~720 lines

Copy from source. Key classes:
- `L2RegressionMomentum` - MLP that learns momentum transformation
- `ContinuumMemoryState` - Per-parameter state for multi-frequency updates
- `DeepNestedOptimizer` - Main optimizer class

### 2. NEW FILE: `titans_core/opt/meta_trainer.py`

**Location**: `titans_core/opt/meta_trainer.py`
**Size**: ~220 lines

Copy from source. Key classes:
- `UnrolledMetaTrainer` - Full k-step unrolled differentiation
- `SimplifiedMetaTrainer` - Proxy-based meta-learning (tracks loss history)
- `create_meta_trainer()` - Factory function

### 3. MODIFY: `titans_core/opt/__init__.py`

Add these imports and exports:

```python
from .deep_nested_optimizer import DeepNestedOptimizer, L2RegressionMomentum, ContinuumMemoryState
from .meta_trainer import UnrolledMetaTrainer, SimplifiedMetaTrainer, create_meta_trainer

__all__ = [
    # New unified optimizer (recommended)
    "DeepNestedOptimizer",
    "L2RegressionMomentum",
    "ContinuumMemoryState",
    # Meta-learning
    "UnrolledMetaTrainer",
    "SimplifiedMetaTrainer",
    "create_meta_trainer",
    # ... existing exports ...
]
```

---

## BCTrainer Integration Changes

### File: `rl_titan_trader/train/bc.py` (or equivalent)

#### Change 1: Optimizer Creation (~line 368-398)

Replace the simple Adam creation with:

```python
# Optimizer
lr = bc_config.get('lr', 0.0003)
optimizer_type = bc_config.get('optimizer_type', 'adam')

if optimizer_type == 'nested':
    # DeepNestedOptimizer: Complete Nested Learning optimizer
    from titans_core.opt import DeepNestedOptimizer
    nested_config = bc_config.get('nested_optimizer', {})
    self.optimizer = DeepNestedOptimizer(
        model=model,
        base_lr=lr,
        meta_lr=nested_config.get('meta_lr', 1e-4),
        k_unroll=nested_config.get('k_unroll', 5),
        cms_frequencies=nested_config.get('cms_frequencies', [1, 10, 100]),
        momentum_hidden_dim=nested_config.get('momentum_hidden_dim', 64),
        controller_hidden_dim=nested_config.get('controller_hidden_dim', 32),
        use_gradient_checkpointing=nested_config.get('use_gradient_checkpointing', True),
        mode=nested_config.get('mode', 'simple'),
        meta_update_freq=nested_config.get('meta_update_freq', 100),
        weight_decay=nested_config.get('weight_decay', 0.0),
        max_grad_norm=1.0,
    )
    self.use_nested_optimizer = True
    print(f"[BCTrainer] Using DeepNestedOptimizer (mode={nested_config.get('mode', 'simple')})")
else:
    # Standard Adam optimizer
    self.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    self.use_nested_optimizer = False
```

#### Change 2: Scheduler Skip for Nested Optimizer

After scheduler creation, add:

```python
# Skip scheduler for nested optimizer (NestedController handles LR adaptation)
if self.use_nested_optimizer:
    print("[BCTrainer] Skipping LR scheduler (NestedController handles LR)")
    self.scheduler = None
elif self.warmup_steps > 0:
    # ... existing scheduler code ...
```

#### Change 3: Optimizer Step in Training Loop

Replace the optimizer step:

```python
# Backward pass
loss_dict['total'].backward()

# Optimizer step
# Note: DeepNestedOptimizer handles its own gradient clipping
if self.use_nested_optimizer:
    # Pass loss value for meta-learning
    self.optimizer.step(loss_dict['total'].item())
    grad_norm = torch.tensor(0.0)  # Already clipped internally
else:
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()
```

#### Change 4: WandB Logging

Update learning rate and add nested optimizer metrics:

```python
# Get learning rate (different API for nested vs standard optimizer)
if self.use_nested_optimizer:
    current_lr = self.optimizer.get_effective_lrs()[0]
else:
    current_lr = self.optimizer.param_groups[0]['lr']

bc_metrics = {
    # ... existing metrics ...
    'bc/learning_rate': current_lr,
}

# Add nested optimizer metrics if using DeepNestedOptimizer
if self.use_nested_optimizer:
    lr_mults = self.optimizer.get_lr_multipliers()
    bc_metrics.update({
        'nested/lr_multiplier_core': to_float(lr_mults[0]),
        'nested/lr_multiplier_memory': to_float(lr_mults[1]) if len(lr_mults) > 1 else 0.0,
        'nested/meta_loss': to_float(self.optimizer.last_meta_loss or 0.0),
        'nested/controller_grad_norm': to_float(self.optimizer.controller_grad_norm),
    })
    mom_stats = self.optimizer.get_momentum_stats()
    if mom_stats:
        bc_metrics['nested/momentum_avg_norm'] = to_float(mom_stats.get('momentum_avg_norm', 0.0))
```

---

## Config Schema

### Add to your YAML config (e.g., `configs/crypto_parquet.yaml`)

```yaml
bc:
  # ... existing bc config ...

  # =============================================================================
  # Optimizer Configuration (Nested Learning)
  # =============================================================================
  # Options: 'adam' (default) or 'nested' (DeepNestedOptimizer)
  optimizer_type: adam  # Change to 'nested' to enable

  # DeepNestedOptimizer configuration (only used if optimizer_type: nested)
  nested_optimizer:
    meta_lr: 0.0001             # Learning rate for meta-optimizer
    k_unroll: 5                 # Unroll steps for meta-learning
    cms_frequencies: [1, 10, 100]  # CMS update frequencies
    momentum_hidden_dim: 64     # MomentumMLP hidden dimension
    controller_hidden_dim: 32   # NestedController hidden dimension
    use_gradient_checkpointing: true
    mode: simple                # 'simple' or 'explicit'
    meta_update_freq: 100       # Meta-update frequency in simple mode
    weight_decay: 0.0
```

---

## Dependencies

The DeepNestedOptimizer depends on existing files in `titans_core/opt/`:

- `nested_controller.py` - NestedController class
- `param_groups.py` - `group_titans_params()`, `infer_param_depth()`

Make sure these exist in the standalone repo.

---

## Unit Tests

Copy the test file to verify the implementation:

**Source**: `tests/unit/test_deep_nested_optimizer.py`
**Tests**: 18 tests covering all components

Run with:
```bash
python -m pytest tests/unit/test_deep_nested_optimizer.py -v
```

---

## Quick Verification

After porting, verify with:

```python
from titans_core.opt import DeepNestedOptimizer
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return {'loss': self.fc(x).sum()}

model = SimpleModel()
optimizer = DeepNestedOptimizer(model, base_lr=1e-3, mode='simple')

# Test step
x = torch.randn(4, 10)
out = model(x)
out['loss'].backward()
result = optimizer.step(out['loss'].item())

print(f"Step: {result['global_step']}")
print(f"LR multipliers: {result['lr_multipliers']}")
# Should see multipliers != 1.0, showing controller is active
```

---

## Architecture Summary

```
DeepNestedOptimizer
├── base_optimizer (AdamW)          # Actual parameter updates
├── momentum_mlp (L2RegressionMomentum)  # Learned gradient compression
├── controller (NestedController)   # Learned LR multipliers
├── state (Dict[Tensor, ContinuumMemoryState])  # Per-param CMS state
├── simplified_meta_trainer         # Proxy-based meta-learning
└── unrolled_meta_trainer          # Full k-step unrolling (explicit mode)
```

**Dual Mode API:**
- `mode='simple'`: Automatic meta-updates every `meta_update_freq` steps
- `mode='explicit'`: Manual `optimizer.meta_update(val_batch)` calls

---

## Key Fixes Over Original Implementation

The original `ContinuumOptimizer` had these issues that are now fixed:

| Issue | Original | Fixed |
|-------|----------|-------|
| Meta-objective | `(mult - 1.0)²` (useless regularization) | Proxy loss tracking improvement |
| MomentumMLP | Never trained (frozen weights) | Trained via meta-optimizer |
| CMS | `requires_grad` hack | Proper per-level state management |
| Training use | Not integrated | Full BCTrainer integration |

---

## Logging to WandB

When enabled, you'll see these metrics in WandB:
- `nested/lr_multiplier_core` - LR multiplier for core params
- `nested/lr_multiplier_memory` - LR multiplier for memory params
- `nested/meta_loss` - Meta-learning loss
- `nested/controller_grad_norm` - Controller gradient norm
- `nested/momentum_avg_norm` - Average momentum buffer norm
