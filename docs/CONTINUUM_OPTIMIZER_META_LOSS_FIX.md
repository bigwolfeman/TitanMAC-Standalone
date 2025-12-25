# ContinuumOptimizer Meta-Loss Fix

**Date:** 2024-12-24
**Issue:** LR multiplier collapses to min_lr_mult (0.1) after ~10 steps and never recovers

## The Bug

The original `ContinuumOptimizer` used pure regularization as its meta-loss:

```python
# BROKEN - Pure regularization with NO training signal
meta_loss = ((multipliers - 1.0) ** 2).mean()
```

### Why This Fails

1. **No Training Signal**: The loss only penalizes deviation from 1.0, with zero information about whether the multiplier actually helped training.

2. **Gradient Points Downward**: The gradient `d(loss)/d(mult) = 2*(mult - 1.0)` creates runaway behavior toward the minimum.

3. **Phase Change at Step ~10**:
   - Steps 1-9: Controller not yet updated, multiplier stays at init (~1.0)
   - Step 10: First controller update, regularization gradient kicks in
   - Steps 11+: Gradient accumulates, multiplier collapses to floor (0.1)

4. **No Recovery**: Once clamped at min_lr_mult, there's no signal to increase it.

## The Fix

Wire `SimplifiedMetaTrainer` (already used by `DeepNestedOptimizer`) into `ContinuumOptimizer`:

```python
# In __init__:
from .meta_trainer import SimplifiedMetaTrainer
self.meta_trainer = SimplifiedMetaTrainer(
    window_size=100,
    improvement_threshold=0.001,
)

# In _update_controller:
# Record step for loss-improvement tracking
self.meta_trainer.record_step(
    loss=loss_value,
    multipliers=multipliers.detach(),
    momentum_norm=0.0,
)

# Use loss-improvement-based proxy loss instead of pure regularization
meta_loss = self.meta_trainer.compute_proxy_loss(
    current_multipliers=multipliers,
    current_loss=loss_value,
)
```

## How SimplifiedMetaTrainer Works

1. **First 10 steps**: Falls back to regularization (building history)
2. **After 10 steps**: Computes `recent_improvement = loss_history[0] - loss_history[-1]`
   - **If improving** (improvement > threshold): Reward staying near current successful multipliers
   - **If stagnating**: Add exploration bonus to encourage change

This provides actual training signal:
- "Loss is decreasing with current multipliers" → keep them
- "Loss is stagnating" → try something different

## Comparison of Meta-Loss Approaches

| Approach | Meta-Loss | Signal Type | Works? |
|----------|-----------|-------------|--------|
| Original ContinuumOptimizer | `((mult - 1.0)²)` | Regularization only | No |
| SimplifiedMetaTrainer | Loss improvement correlation | Training dynamics | Yes |
| UnrolledMetaTrainer | L_{t+k} (val loss after k steps) | True outer-loop | Yes (expensive) |

## Files Changed

- `titans_core/opt/continuum_optimizer.py`: Added SimplifiedMetaTrainer integration

## Related

- `titans_core/opt/meta_trainer.py`: Contains SimplifiedMetaTrainer and UnrolledMetaTrainer
- `titans_core/opt/deep_nested_optimizer.py`: Uses these trainers correctly
- Nested Learning paper (NeurIPS 2025): Describes proper meta-loss formulation
