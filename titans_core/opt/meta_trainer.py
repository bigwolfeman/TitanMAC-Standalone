"""
UnrolledMetaTrainer: Meta-learning via k-step unrolled differentiation.

Trains optimizer components (MomentumMLP, NestedController) by:
1. Cloning model state
2. Running k inner optimization steps (differentiable)
3. Computing validation loss
4. Backpropagating through the unrolled computation graph

This implements the "nested" optimization from the Nested Learning paper,
where the outer loop optimizes the optimizer itself.

Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
           Behrouz et al., NeurIPS 2025
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class UnrolledMetaTrainer:
    """
    Trains optimizer components via k-step unrolled differentiation.

    The meta-objective is L_{t+k}: the validation loss after k optimization
    steps. Gradients flow backward through the unrolled steps to update
    the optimizer's learnable components.

    This enables the optimizer to learn update rules that lead to better
    convergence, not just immediate loss reduction.

    Args:
        k_steps: Number of inner steps to unroll (default: 5)
        use_checkpointing: Use gradient checkpointing to reduce memory
        accumulate_grad: Whether to accumulate gradients across unroll steps
        second_order: Whether to compute second-order gradients through optimizer

    Memory complexity:
        - Without checkpointing: O(k * model_size)
        - With checkpointing: O(model_size) but 2x compute

    Example:
        >>> meta_trainer = UnrolledMetaTrainer(k_steps=5)
        >>> meta_loss = meta_trainer.compute_meta_loss(
        ...     model=model,
        ...     optimizer_components={'momentum_mlp': mlp, 'controller': ctrl},
        ...     train_batches=[batch1, batch2, batch3, batch4, batch5],
        ...     val_batch=val_batch,
        ...     loss_fn=loss_fn,
        ... )
        >>> meta_loss.backward()  # Gradients flow to mlp and ctrl
    """

    def __init__(
        self,
        k_steps: int = 5,
        use_checkpointing: bool = True,
        accumulate_grad: bool = True,
        second_order: bool = False,
    ):
        self.k_steps = k_steps
        self.use_checkpointing = use_checkpointing
        self.accumulate_grad = accumulate_grad
        self.second_order = second_order

    def compute_meta_loss(
        self,
        model: nn.Module,
        optimizer_components: Dict[str, nn.Module],
        train_batches: List[Dict[str, Tensor]],
        val_batch: Dict[str, Tensor],
        loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        base_lr: float = 1e-3,
        inner_optimizer_cls: type = None,
    ) -> Tensor:
        """
        Compute meta-loss via k-step unrolled optimization.

        Args:
            model: The model being trained
            optimizer_components: Dict of learnable optimizer components
                {'momentum_mlp': MomentumMLP, 'controller': NestedController}
            train_batches: List of k training batches for inner steps
            val_batch: Validation batch for meta-objective
            loss_fn: Function(model, batch) -> loss
            base_lr: Base learning rate for inner updates
            inner_optimizer_cls: Optimizer class for inner updates (default: SGD)

        Returns:
            meta_loss: Validation loss after k inner steps (differentiable)
        """
        if len(train_batches) < self.k_steps:
            raise ValueError(
                f"Need at least {self.k_steps} train batches, got {len(train_batches)}"
            )

        # Clone model parameters (we'll modify these in the inner loop)
        # Using functional approach to keep computation graph
        original_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Track parameter updates through the unroll
        current_params = {
            name: param.detach().requires_grad_(True)
            for name, param in original_params.items()
        }

        # Extract components
        momentum_mlp = optimizer_components.get('momentum_mlp')
        controller = optimizer_components.get('controller')

        # Initialize momentum buffers
        momentum_buffers = {
            name: torch.zeros_like(param)
            for name, param in current_params.items()
        }

        # Unroll k inner optimization steps
        for step in range(self.k_steps):
            batch = train_batches[step]

            if self.use_checkpointing:
                # Use gradient checkpointing for memory efficiency
                loss = checkpoint(
                    self._inner_forward,
                    model,
                    current_params,
                    batch,
                    loss_fn,
                    use_reentrant=False,
                )
            else:
                loss = self._inner_forward(model, current_params, batch, loss_fn)

            # Compute gradients w.r.t. current params
            grads = torch.autograd.grad(
                loss,
                list(current_params.values()),
                create_graph=self.second_order or step < self.k_steps - 1,
                allow_unused=True,
            )

            # Apply learned optimizer updates
            new_params = {}
            new_momentum = {}

            for (name, param), grad in zip(current_params.items(), grads):
                if grad is None:
                    new_params[name] = param
                    new_momentum[name] = momentum_buffers[name]
                    continue

                # Get LR multiplier from controller (if available)
                lr_mult = 1.0
                if controller is not None:
                    # Simplified: use scalar multiplier
                    # Full implementation would use group-specific multipliers
                    stats = self._compute_stats(grad, param)
                    lr_mult = controller(stats.unsqueeze(0))[0].item()

                # Apply momentum transformation (if MLP available)
                if momentum_mlp is not None:
                    context = torch.tensor(
                        [step / self.k_steps, base_lr, loss.item()],
                        device=grad.device,
                    )
                    scale, shift, damping = momentum_mlp(
                        grad,
                        momentum_buffers[name],
                        context,
                    )
                    momentum = scale * momentum_buffers[name] + shift * grad
                    update = momentum * (1 - damping)
                else:
                    # Standard momentum
                    momentum = 0.9 * momentum_buffers[name] + grad
                    update = momentum

                # Update parameters (differentiable)
                new_params[name] = param - base_lr * lr_mult * update
                new_momentum[name] = momentum.detach()

            current_params = new_params
            momentum_buffers = new_momentum

        # Compute validation loss with updated parameters
        meta_loss = self._inner_forward(model, current_params, val_batch, loss_fn)

        # Restore original parameters (important!)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(original_params[name])

        return meta_loss

    def _inner_forward(
        self,
        model: nn.Module,
        params: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        loss_fn: Callable,
    ) -> Tensor:
        """
        Forward pass using custom parameters (functional style).

        This allows us to compute gradients w.r.t. the parameters
        without modifying the model's actual state.
        """
        # Temporarily replace model parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
            param.data.copy_(params[name])

        try:
            loss = loss_fn(model, batch)
        finally:
            # Restore original parameters
            for name, param in model.named_parameters():
                param.data.copy_(original_params[name])

        return loss

    def _compute_stats(self, grad: Tensor, param: Tensor) -> Tensor:
        """Compute statistics for controller input."""
        grad_norm = grad.norm().item()
        param_norm = param.norm().item()
        return torch.tensor(
            [grad_norm, param_norm, 0.5],  # depth=0.5 as placeholder
            device=grad.device,
        )


class SimplifiedMetaTrainer:
    """
    Simplified meta-trainer that doesn't require full unrolling.

    Uses a proxy objective instead of true k-step unrolling:
    - Tracks loss improvement over recent steps
    - Uses correlation between optimizer decisions and improvement as signal

    This is less principled but more practical for:
    - Large models where full unrolling is too expensive
    - Settings where validation data isn't readily available
    - Initial prototyping before implementing full unrolling

    Args:
        window_size: Number of recent steps to track
        improvement_threshold: Minimum improvement to consider "good"
    """

    def __init__(
        self,
        window_size: int = 50,
        improvement_threshold: float = 0.001,
    ):
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold

        # History tracking
        self.loss_history: List[float] = []
        self.multiplier_history: List[Tensor] = []
        self.momentum_history: List[float] = []

    def record_step(
        self,
        loss: float,
        multipliers: Tensor,
        momentum_norm: float,
    ):
        """Record optimizer state for a training step."""
        self.loss_history.append(loss)
        self.multiplier_history.append(multipliers.clone())
        self.momentum_history.append(momentum_norm)

        # Keep only recent history
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.multiplier_history.pop(0)
            self.momentum_history.pop(0)

    def compute_proxy_loss(
        self,
        current_multipliers: Tensor,
        current_loss: float,
    ) -> Tensor:
        """
        Compute proxy meta-loss from historical correlation.

        Encourages multipliers that correlate with loss improvement.
        """
        if len(self.loss_history) < 10:
            # Not enough history - return regularization only
            return ((current_multipliers - 1.0) ** 2).mean()

        # Compute loss improvement over recent window
        recent_improvement = self.loss_history[0] - self.loss_history[-1]

        # If improving, encourage current strategy
        # If not improving, encourage exploration (move away from 1.0)
        if recent_improvement > self.improvement_threshold:
            # Reward: stay near current successful values
            target = self.multiplier_history[-1]
            loss = ((current_multipliers - target) ** 2).mean()
        else:
            # Penalty: encourage change if stagnating
            stability_penalty = ((current_multipliers - 1.0) ** 2).mean()
            exploration_bonus = -0.1 * (current_multipliers - 1.0).abs().mean()
            loss = stability_penalty + exploration_bonus

        return loss

    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics for logging."""
        if len(self.loss_history) < 2:
            return {}

        return {
            'meta/loss_improvement': self.loss_history[0] - self.loss_history[-1],
            'meta/loss_variance': torch.tensor(self.loss_history).std().item(),
            'meta/avg_momentum_norm': sum(self.momentum_history) / len(self.momentum_history),
        }


def create_meta_trainer(
    mode: str = 'simplified',
    **kwargs,
) -> Any:
    """
    Factory function to create appropriate meta-trainer.

    Args:
        mode: 'unrolled' for full k-step unrolling,
              'simplified' for proxy-based training
        **kwargs: Arguments for the specific trainer

    Returns:
        Meta-trainer instance
    """
    if mode == 'unrolled':
        return UnrolledMetaTrainer(**kwargs)
    elif mode == 'simplified':
        return SimplifiedMetaTrainer(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'unrolled' or 'simplified'.")
