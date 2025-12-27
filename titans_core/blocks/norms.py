"""
Normalization layers for Titan-MAC.

This module provides RMSNorm (Root Mean Square Normalization) for stable
training in the Titan architecture.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    More stable than LayerNorm for transformer architectures as it doesn't
    depend on batch statistics (no mean centering).

    Formula:
        x_norm = x / RMS(x) * scale
        where RMS(x) = sqrt(mean(x²) + eps)

    Args:
        dim: Embedding dimension
        eps: Epsilon for numerical stability (default: 1e-5)

    Shape:
        Input: [..., dim]
        Output: [..., dim]

    Example:
        >>> norm = RMSNorm(640)
        >>> x = torch.randn(2, 512, 640)
        >>> x_norm = norm(x)
        >>> x_norm.shape
        torch.Size([2, 512, 640])
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        # Compute RMS: sqrt(mean(x²) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_norm = x / rms * self.scale

        return x_norm
