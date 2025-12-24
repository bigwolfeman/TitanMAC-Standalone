# Nested Optimizer Improvement Ideas

**Date**: December 19, 2025
**Updated**: December 20, 2025
**Status**: Ideas for future implementation

> **See Also**:
> - `GPU_PROFILING_REPORT.md` - Performance analysis findings
> - `PERFORMANCE_OPTIMIZATIONS.md` - Implemented optimizations (64% CUDA reduction)
> - `OPTIMIZER_QUICK_REFERENCE.md` - Parameter reference and usage examples

## Current Training Signals

| Component | Input | Output | Training Signal |
|-----------|-------|--------|-----------------|
| **NestedController** | [grad_norm, param_norm, depth] | LR multipliers | Proxy loss: "did loss improve when multiplier was X?" |
| **DirectUpdateMLP** | [log_g, sign_g, log_m, sign_m] | Per-element update | Surrogate: cosine(output, grad) + magnitude match |

**The problem**: These are very indirect signals. The controller doesn't know *why* loss improved - many things affect it. The MLP is basically learning to mimic SGD with slight adjustments.

---

## High-Leverage Opportunities

### 1. Hindsight Optimal LR (Oracle for Optimizer)

Just like trading oracle labels future returns, we could label "what LR would have been optimal in hindsight":

```python
# After every K steps, compute hindsight-optimal LR
def compute_hindsight_lr(cached_grads, param_snapshots, loss_trajectory):
    """Try different LRs on cached grads, see which minimized loss most."""
    best_lr_mult = grid_search([0.1, 0.5, 1.0, 1.5, 2.0], cached_grads, ...)
    return best_lr_mult  # Supervision signal!
```

**Implementation sketch**:
- Cache gradients for last K steps
- Every K steps, replay with different LR multipliers
- Label the multiplier that worked best
- Train controller with cross-entropy on this label

### 2. Validation-Aware Controller

Controller only sees training stats. Feed it validation signals:

```python
# Current controller input: [grad_norm, param_norm, depth]
# Enhanced: [grad_norm, param_norm, depth, val_loss_delta, train_val_gap]
```

This lets it detect overfitting and adjust LR accordingly.

**New features**:
- `val_loss_delta`: Is validation loss improving or degrading?
- `train_val_gap`: Gap between train and val loss (overfitting indicator)
- `val_loss_ema`: Smoothed validation loss trend

### 3. Optimizer Scratchpad/Memory

Give the controller persistent state:

```python
class NestedControllerWithMemory(nn.Module):
    def __init__(self, hidden_dim=32, scratchpad_dim=16, ...):
        super().__init__()
        self.scratchpad = nn.Parameter(torch.zeros(scratchpad_dim))
        self.memory_update = nn.Linear(hidden_dim + scratchpad_dim, scratchpad_dim)
        self.net = ...  # existing network

    def forward(self, stats):
        # Read from scratchpad
        combined = torch.cat([stats.flatten(), self.scratchpad])

        # Compute output
        hidden = self.encoder(combined)
        multipliers = self.output_head(hidden)

        # Write to scratchpad (what patterns led to good updates?)
        self.scratchpad.data = torch.tanh(self.memory_update(hidden))

        return multipliers
```

**What the scratchpad could learn**:
- Which gradient patterns preceded good/bad updates
- Temporal correlations in learning dynamics
- Phase detection (early training vs convergence)

### 4. Trading-Specific Signals (for Fin-TitanMAC)

The optimizer could see task-specific information:

| Signal | Why It Helps |
|--------|--------------|
| Oracle accuracy per horizon | "7d head stuck at 0.50 AUC - increase its LR" |
| Loss per horizon (not just total) | "Short horizons converged, focus on long" |
| Gradient magnitude per head | "14d gradients are tiny - signal or vanishing?" |
| Simulated PnL | "Model losing money - something's wrong" |

```python
# Enhanced controller input for trading
trading_stats = {
    'horizon_aucs': [0.78, 0.64, 0.66, 0.63, 0.60, 0.50, 0.50],  # 10m to 14d
    'horizon_losses': [...],
    'horizon_grad_norms': [...],
    'oracle_alignment': correlation(pred, oracle),
    'market_regime': regime_classifier(recent_returns),
}
```

### 5. Gradient Curvature (Cheap Hessian Proxy)

The loss landscape curvature tells us optimal step size:

```python
def estimate_curvature(grad_t, grad_t_minus_1, param_delta):
    """Approximate Hessian-vector product from gradient differences."""
    # Finite difference approximation of Hv where v = param_delta
    hessian_approx = (grad_t - grad_t_minus_1) / (param_delta.norm() + 1e-8)

    # High curvature → smaller LR, low curvature → larger LR
    curvature_factor = 1.0 / (hessian_approx.abs().mean() + 1.0)
    return curvature_factor
```

**Per-group curvature**:
- Core params might have different curvature than memory params
- Controller could learn curvature → LR mapping

### 6. Meta-Learning on Actual Improvement

Instead of proxy losses, use actual parameter improvement:

```python
def compute_actual_improvement_loss(
    param_before, param_after, val_loss_before, val_loss_after
):
    """Did this update actually help on validation?"""
    improvement = val_loss_before - val_loss_after
    param_delta = param_after - param_before

    # Reward updates that led to improvement
    # Penalize updates that hurt validation
    return -improvement * param_delta.norm()
```

---

## Priority Ranking for Trading Model

1. **Per-horizon loss feedback** (Quick win)
   - Already compute these losses
   - Just route them to controller input
   - Lets optimizer know which heads need more/less LR

2. **Oracle alignment signal** (Medium effort)
   - Compute correlation(model_pred, oracle_label) per horizon
   - Direct feedback on what matters for trading

3. **Optimizer scratchpad** (Higher effort)
   - Requires architecture change
   - But could learn temporal patterns in training dynamics

4. **Hindsight optimal LR** (Highest effort)
   - Requires gradient caching and replay
   - Most principled but computationally expensive

---

## Implementation Notes

### Modifying NestedController

```python
class EnhancedNestedController(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        n_groups: int = 2,
        n_horizons: int = 7,  # For trading
        use_scratchpad: bool = False,
        scratchpad_dim: int = 16,
    ):
        super().__init__()

        # Base features: [grad_norm, param_norm, depth] per group
        base_features = 3 * n_groups

        # Horizon features: [loss, auc, grad_norm] per horizon
        horizon_features = 3 * n_horizons if n_horizons > 0 else 0

        # Validation features: [val_loss_delta, train_val_gap]
        val_features = 2

        # Scratchpad
        self.use_scratchpad = use_scratchpad
        self.scratchpad_dim = scratchpad_dim if use_scratchpad else 0

        total_input = base_features + horizon_features + val_features + self.scratchpad_dim

        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_groups),
        )

        if use_scratchpad:
            self.scratchpad = nn.Parameter(torch.zeros(scratchpad_dim))
            self.memory_gate = nn.Linear(hidden_dim, scratchpad_dim)
```

### Passing Horizon Stats from BCTrainer

```python
# In BCTrainer.train_step():
if self.use_nested_optimizer:
    # Collect horizon-specific stats
    horizon_stats = {
        'losses': [loss_dict.get(f'horizon_{h}_loss', 0) for h in horizons],
        'aucs': [self.horizon_aucs.get(h, 0.5) for h in horizons],
        'grad_norms': self._compute_horizon_grad_norms(),
    }

    # Pass to optimizer
    self.optimizer.step(loss.item(), horizon_stats=horizon_stats)
```

---

## References

- Andrychowicz et al. 2016: "Learning to learn by gradient descent by gradient descent"
- Behrouz et al. 2025: "Nested Learning: The Illusion of Deep Learning Architectures"
- Metz et al. 2022: "VeLO: Training Versatile Learned Optimizers"
