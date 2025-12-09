"""MLP block for Titan-MAC."""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Simple 2-layer MLP with GELU activation.

    Args:
        d_model: Model dimension (input/output)
        d_ff: Feed-forward dimension (hidden layer)
        dropout: Dropout probability (default: 0.0)

    Shape:
        Input: [B, T, d_model]
        Output: [B, T, d_model]

    Example:
        >>> mlp = MLPBlock(d_model=128, d_ff=512)
        >>> x = torch.randn(2, 64, 128)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([2, 64, 128])
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = dropout

        # Two-layer MLP
        # Up projection: d_model → d_ff
        self.fc1 = nn.Linear(d_model, d_ff)

        # Down projection: d_ff → d_model
        self.fc2 = nn.Linear(d_ff, d_model)

        # Activation
        self.activation = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP block.

        Args:
            x: Input tensor [B, T, d_model]

        Returns:
            Output tensor [B, T, d_model]
        """
        # Up projection
        hidden = self.fc1(x)  # [B, T, d_ff]

        # Activation
        hidden = self.activation(hidden)

        # Down projection
        output = self.fc2(hidden)  # [B, T, d_model]
        output = self.dropout(output)

        return output
