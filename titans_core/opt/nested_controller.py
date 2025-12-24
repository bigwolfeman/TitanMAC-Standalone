"""
NestedController: Small MLP for learning rate modulation.

Maps per-group gradient statistics to LR scaling factors for
multi-timescale learning in the DeepNestedOptimizer.

Ported from TitanMAC for MoE model training.
"""

import torch
import torch.nn as nn


class NestedController(nn.Module):
    """
    Small MLP that predicts learning rate multipliers.

    Maps per-group gradient statistics to LR scaling factors
    for multi-timescale learning.

    Args:
        hidden_dim: Controller hidden dimension (default: 32)
        min_lr_mult: Minimum LR multiplier (default: 0.1)
        max_lr_mult: Maximum LR multiplier (default: 2.0)
        n_groups: Number of parameter groups (default: 2)

    Input:
        stats: [n_groups, 3] - (grad_norm, param_norm, avg_depth)

    Output:
        multipliers: [n_groups] - LR multipliers in [min, max]

    Example:
        >>> controller = NestedController(hidden_dim=32)
        >>> stats = torch.tensor([[0.5, 1.2, 0.3], [0.3, 0.8, 0.5]])
        >>> multipliers = controller(stats)
        >>> print(multipliers)  # tensor([1.23, 0.87])
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        min_lr_mult: float = 0.1,
        max_lr_mult: float = 2.0,
        n_groups: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_lr_mult = min_lr_mult
        self.max_lr_mult = max_lr_mult
        self.n_groups = n_groups

        # Input: [n_groups, 3] features
        # Output: [n_groups] multipliers
        input_dim = 3  # grad_norm, param_norm, avg_depth

        # Build MLP with configurable depth
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))  # Output 1 value per group
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """
        Predict LR multipliers from gradient statistics.

        Args:
            stats: Per-group statistics [n_groups, 3]
                   Format: [grad_norm, param_norm, avg_depth]

        Returns:
            LR multipliers [n_groups] in [min_lr_mult, max_lr_mult]

        Note:
            Output is clamped to [min_lr_mult, max_lr_mult] by design.
            Uses branchless NaN handling to avoid GPU sync from .item() calls.
        """
        # Default multipliers (1.0) - used as fallback for NaN
        default_multipliers = torch.ones(
            self.n_groups, device=stats.device, dtype=stats.dtype
        )

        # Check for NaN/Inf in input (keep as tensor, no .item()!)
        input_nan_mask = torch.isnan(stats) | torch.isinf(stats)
        has_input_nan = input_nan_mask.any()

        # Replace NaN/Inf in input with safe values before MLP forward
        # This prevents NaN propagation through the network
        safe_stats = torch.where(input_nan_mask, torch.ones_like(stats), stats)

        # Process through MLP
        # stats: [n_groups, 3] -> net output: [n_groups, 1]
        raw_output = self.net(safe_stats)  # [n_groups, 1]

        # Squeeze to [n_groups]
        raw_output = raw_output.squeeze(-1)  # [n_groups]

        # Apply sigmoid to get values in [0, 1]
        normalized = torch.sigmoid(raw_output)

        # Scale to [min_lr_mult, max_lr_mult]
        multipliers = (
            self.min_lr_mult + normalized * (self.max_lr_mult - self.min_lr_mult)
        )

        # Ensure bounds (redundant with sigmoid, but explicit)
        multipliers = torch.clamp(multipliers, self.min_lr_mult, self.max_lr_mult)

        # Check for NaN/Inf in output (keep as tensor, no .item()!)
        output_nan_mask = torch.isnan(multipliers) | torch.isinf(multipliers)
        has_output_nan = output_nan_mask.any()

        # Branchless NaN handling: use torch.where to select defaults where needed
        # If any input was NaN, use all defaults (input NaN affects all outputs)
        # If specific outputs are NaN, replace just those
        multipliers = torch.where(output_nan_mask, default_multipliers, multipliers)

        # If input had NaN, return all defaults
        # Expand has_input_nan to match multipliers shape for torch.where
        multipliers = torch.where(
            has_input_nan.expand(self.n_groups),
            default_multipliers,
            multipliers
        )

        return multipliers
