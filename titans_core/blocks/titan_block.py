"""Titan block combining attention and MLP with pre-norm architecture."""

import torch
import torch.nn as nn
from typing import Optional

from ..config import TitanMACConfig
from .norms import RMSNorm
from .mlp import MLPBlock
from ..attn.windowed_attention import WindowedAttention


class TitanBlock(nn.Module):
    """
    Single Titan-MAC transformer block.

    Architecture (pre-norm):
        1. Attention: x = x + Attention(RMSNorm(x))
        2. MLP: x = x + MLP(RMSNorm(x))

    Args:
        config: TitanMACConfig with architecture parameters

    Shape:
        Input: [B, T, d_model]
        Output: [B, T, d_model]

    Example:
        >>> from titans_core.config import TitanMACConfig
        >>> config = TitanMACConfig(d_model=640, n_heads=10)
        >>> block = TitanBlock(config)
        >>> x = torch.randn(2, 512, 640)
        >>> out = block(x)
    """

    def __init__(self, config: TitanMACConfig):
        super().__init__()

        self.config = config

        # Normalization layers (pre-norm architecture)
        self.norm_attn = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm_mlp = RMSNorm(config.d_model, eps=config.norm_eps)

        # Attention block
        self.attention = WindowedAttention(config)

        # MLP block
        self.mlp = MLPBlock(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Titan block.

        Args:
            x: Input tensor [B, T, d_model]
            attn_mask: Optional attention mask

        Returns:
            Output tensor [B, T, d_model]

        Process:
            1. Attention with residual
            2. MLP with residual
        """
        # Attention with pre-norm and residual
        attn_input = self.norm_attn(x)
        attn_out = self.attention(attn_input, attn_mask=attn_mask)
        x = x + attn_out

        # MLP with pre-norm and residual
        mlp_input = self.norm_mlp(x)
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out

        return x
