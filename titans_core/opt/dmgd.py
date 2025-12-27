"""
Deep Momentum Gradient Descent (DMGD) from Nested Learning paper.

This module implements learned momentum via small MLPs that replace
the linear momentum update rule in standard optimizers.

Standard Momentum: v_t = β * v_{t-1} + g_t
DMGD: v_t = MLP_φ(v_{t-1}, g_t, context)

The MLP learns to predict optimal momentum transformations based on
gradient statistics and training context.

Version: 1.0.0
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MomentumMLP(nn.Module):
    """
    Learned momentum computation replacing linear momentum.

    A small 2-layer MLP that takes statistics about previous momentum,
    current gradients, and training context to produce momentum
    transformation parameters.

    Architecture:
        - Input: [v_stats, g_stats, context] = 9 features
        - Hidden: 2 layers with SiLU activation
        - Output: [scale, shift, damping] = 3 parameters

    Output semantics:
        - scale: Momentum retention factor in [0, 2]
        - shift: Gradient mixing factor in [-1, 1]
        - damping: Damping factor in [0, 1]

    Args:
        hidden_dim: Hidden layer dimension (default: 64)
        input_dim: Input dimension (default: 9 = 3 stats × 3 sources)

    Example:
        >>> mlp = MomentumMLP(hidden_dim=64)
        >>> v_stats = torch.tensor([0.1, 0.2, 1.0])  # mean, std, norm
        >>> g_stats = torch.tensor([0.01, 0.1, 0.5])
        >>> context = torch.tensor([0.1, 0.001, 0.5])  # step_norm, lr, loss
        >>> scale, shift, damping = mlp(v_stats, g_stats, context)
    """

    def __init__(self, hidden_dim: int = 64, input_dim: int = 9):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Output: scale, shift, damping
        )

        # Initialize output layer with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable initial behavior.

        Initial outputs should be:
            - scale ≈ 0.9 (retain most momentum)
            - shift ≈ 1.0 (use gradient like standard SGD)
            - damping ≈ 0.0 (no damping initially)
        """
        with torch.no_grad():
            # Output layer
            output_layer = self.net[-1]
            nn.init.zeros_(output_layer.weight)
            # Bias: set to produce reasonable defaults via sigmoid/tanh
            # sigmoid(0.8) ≈ 0.69 -> scale ≈ 1.38 (will be clamped to 2 max)
            # tanh(1.5) ≈ 0.91 -> shift ≈ 0.91 (uses gradient)
            # sigmoid(-2) ≈ 0.12 -> damping ≈ 0.12 (minimal damping)
            output_layer.bias.data = torch.tensor([0.8, 1.5, -2.0])

    def forward(
        self,
        v_stats: Tensor,
        g_stats: Tensor,
        context: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute momentum transformation parameters.

        Args:
            v_stats: Previous momentum statistics [3] (mean, std, norm)
            g_stats: Current gradient statistics [3] (mean, std, norm)
            context: Context vector [3] (normalized_step, lr, loss)

        Returns:
            scale: Momentum retention factor [0, 2]
            shift: Gradient mixing factor [-1, 1]
            damping: Damping factor [0, 1]
        """
        # Ensure inputs are flattened to 1D
        v_stats = v_stats.view(-1)
        g_stats = g_stats.view(-1)
        context = context.view(-1)

        # Concatenate inputs
        x = torch.cat([v_stats, g_stats, context], dim=-1)

        # Forward through network
        out = self.net(x)

        # Transform outputs to conservative bounded ranges
        # Scale: [0.5, 1.5] instead of [0, 2] to prevent exponential growth
        scale = torch.sigmoid(out[0]) * 1.0 + 0.5  # [0.5, 1.5] range
        shift = torch.tanh(out[1])  # [-1, 1] range
        damping = torch.sigmoid(out[2])  # [0, 1] range

        return scale, shift, damping


class DMGDOptimizer:
    """
    Deep Momentum Gradient Descent Optimizer.

    Replaces standard linear momentum with learned MLP-based momentum.
    Each parameter group can have its own MomentumMLP for specialized
    momentum computation.

    Update formula:
        stats = compute_statistics(v, g, context)
        scale, shift, damping = MomentumMLP(stats)
        v_new = scale * v + shift * g
        p = p - lr * v_new * (1 - damping)

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        n_groups: Number of parameter groups for DMGD MLPs (default: 1)
        hidden_dim: MomentumMLP hidden dimension (default: 64)

    Example:
        >>> optimizer = DMGDOptimizer(model.parameters(), lr=1e-3)
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = model(batch).loss
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        n_groups: int = 1,
        hidden_dim: int = 64,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if n_groups <= 0:
            raise ValueError(f"n_groups must be positive, got {n_groups}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.params = list(params)
        self.lr = lr
        self.n_groups = n_groups
        self.hidden_dim = hidden_dim

        # Create one MomentumMLP per group
        self.momentum_mlps = nn.ModuleList(
            [MomentumMLP(hidden_dim=hidden_dim) for _ in range(n_groups)]
        )

        # Initialize momentum buffers (zeros)
        self.v_buffers: List[Tensor] = []
        for p in self.params:
            self.v_buffers.append(torch.zeros_like(p.data))

        # Training state
        self.global_step = 0
        self._last_loss: Optional[float] = None

    def _compute_stats(self, tensor: Tensor) -> Tensor:
        """Compute statistics for a tensor: [mean, std, norm]."""
        with torch.no_grad():
            mean = tensor.mean().item()
            std = tensor.std().item() if tensor.numel() > 1 else 0.0
            norm = tensor.norm().item()
        return torch.tensor([mean, std, norm], device=tensor.device, dtype=tensor.dtype)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform single optimization step.

        Args:
            closure: Optional closure for re-evaluating loss

        Returns:
            loss: Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Get loss for context
        loss_value = loss if loss is not None else (self._last_loss or 0.0)

        for i, (p, v) in enumerate(zip(self.params, self.v_buffers)):
            if p.grad is None:
                continue

            g = p.grad.data

            # Compute statistics
            v_stats = self._compute_stats(v)
            g_stats = self._compute_stats(g)

            # Context: normalized step, lr, normalized loss
            # Use log(loss+1) to prevent large loss values from destabilizing MLP
            norm_loss = math.log(max(float(loss_value), 1e-8) + 1.0)
            context = torch.tensor(
                [self.global_step / 1000.0, self.lr, norm_loss],
                device=p.device,
                dtype=p.dtype,
            )

            # Get MLP for this parameter's group
            mlp_idx = i % self.n_groups
            mlp = self.momentum_mlps[mlp_idx]

            # Compute learned momentum transformation
            scale, shift, damping = mlp(v_stats, g_stats, context)

            # Clip gradient before computing momentum (prevent explosion)
            g_norm = g.norm()
            max_grad = 10.0
            if g_norm > max_grad:
                g = g * (max_grad / g_norm)

            # Update momentum: v_new = scale * v + shift * g
            v_new = scale.item() * v + shift.item() * g

            # Clip momentum buffer to prevent explosion (critical for stability)
            max_momentum = 100.0  # Reduced from 1e4 for tighter control
            v_new = torch.clamp(v_new, min=-max_momentum, max=max_momentum)

            # Skip update if NaN/Inf detected
            if torch.isnan(v_new).any() or torch.isinf(v_new).any():
                continue

            # Store new momentum
            self.v_buffers[i] = v_new.detach()

            # Compute update and clip it
            update = v_new * (1 - damping.item())
            max_update = 1.0  # Maximum per-parameter update magnitude
            update = torch.clamp(update, min=-max_update, max=max_update)

            # Update parameter: p = p - lr * update
            p.data.add_(update, alpha=-self.lr)

        # Increment step counter
        self.global_step += 1

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all parameters."""
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def set_loss(self, loss_value: float) -> None:
        """Set loss value for context (alternative to closure)."""
        self._last_loss = loss_value

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "momentum_mlps": self.momentum_mlps.state_dict(),
            "v_buffers": [v.clone() for v in self.v_buffers],
            "global_step": self.global_step,
            "lr": self.lr,
            "n_groups": self.n_groups,
            "hidden_dim": self.hidden_dim,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""
        if "momentum_mlps" in state_dict:
            self.momentum_mlps.load_state_dict(state_dict["momentum_mlps"])

        if "v_buffers" in state_dict:
            for i, v in enumerate(state_dict["v_buffers"]):
                if i < len(self.v_buffers):
                    self.v_buffers[i].copy_(v)

        if "global_step" in state_dict:
            self.global_step = state_dict["global_step"]

        if "lr" in state_dict:
            self.lr = state_dict["lr"]

    def get_momentum_stats(self) -> Dict[str, float]:
        """Get statistics about momentum buffers for logging."""
        if not self.v_buffers:
            return {}

        total_norm = sum(v.norm().item() for v in self.v_buffers)
        avg_norm = total_norm / len(self.v_buffers)

        return {
            "momentum_total_norm": total_norm,
            "momentum_avg_norm": avg_norm,
            "global_step": float(self.global_step),
        }

    def to(self, device: torch.device) -> "DMGDOptimizer":
        """Move optimizer to device."""
        self.momentum_mlps = self.momentum_mlps.to(device)
        self.v_buffers = [v.to(device) for v in self.v_buffers]
        return self
