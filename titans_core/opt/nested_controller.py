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
        """
        # Validate input shape
        assert stats.shape[0] == self.n_groups, (
            f"Expected {self.n_groups} groups, got {stats.shape[0]}"
        )
        assert stats.shape[1] == 3, (
            f"Expected 3 features per group, got {stats.shape[1]}"
        )

        # Process each group independently
        # stats: [n_groups, 3]
        # net output: [n_groups, 1]
        raw_output = self.net(stats)  # [n_groups, 1]

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

        return multipliers
