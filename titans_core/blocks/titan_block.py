"""Titan block combining attention and MLP with pre-norm architecture."""

import torch
import torch.nn as nn
from typing import Optional

from ..config import TitanMACConfig
from .norms import RMSNorm
from .mlp import MLPBlock
from ..attn.windowed_attention import WindowedAttention
from ..attn.block_sparse_attention import BlockSparseAttention


class TitanBlock(nn.Module):
    """
    Single Titan-MAC transformer block.

    Architecture (pre-norm):
        1. Attention: x = x + Attention(RMSNorm(x))
        2. MLP: x = x + MLP(RMSNorm(x))

    Attention options:
        - use_block_sparse=False: O(T²) WindowedAttention (legacy)
        - use_block_sparse=True: O(T*w) BlockSparseAttention (paper-faithful)

    MLP options:
        - Dense nn.Linear (default)
        - CMSBlockLinear when mlp_use_block_sparse=True and layer_idx in block_sparse_layers

    Args:
        config: TitanMACConfig with architecture parameters
        layer_idx: Index of this layer in the stack (0-based), used for per-layer config

    Shape:
        Input: [B, T, d_model]
        Output: [B, T, d_model]

    Example:
        >>> from titans_core.config import TitanMACConfig
        >>> config = TitanMACConfig(d_model=640, n_heads=10)
        >>> block = TitanBlock(config, layer_idx=0)
        >>> x = torch.randn(2, 512, 640)
        >>> out = block(x)
    """

    def __init__(self, config: TitanMACConfig, layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Normalization layers (pre-norm architecture)
        self.norm_attn = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm_mlp = RMSNorm(config.d_model, eps=config.norm_eps)

        # Attention block - choose implementation based on config
        if getattr(config, "use_block_sparse", False):
            # Paper-faithful O(T*w) block-sparse attention
            self.attention = BlockSparseAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                window_size=config.window_size,
                block_size=getattr(config, "block_size", 64),
                n_persistent=config.n_persistent,
                dropout=config.attention_dropout,
                causal=config.causal,
            )
        else:
            # Legacy O(T²) windowed attention
            self.attention = WindowedAttention(config)

        # MLP block - potentially with block-sparse layers
        # Determine if this layer should use block-sparse MLP
        use_mlp_block_sparse = self._should_use_block_sparse_mlp(config, layer_idx)
        mlp_density = self._get_layer_density(config, layer_idx)

        self.mlp = MLPBlock(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            use_block_sparse=use_mlp_block_sparse,
            block_sparse_tile_size=getattr(config, "mlp_block_sparse_tile_size", 16),
            block_sparse_density=mlp_density,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    @staticmethod
    def _should_use_block_sparse_mlp(config: TitanMACConfig, layer_idx: int) -> bool:
        """Determine if this layer should use block-sparse MLP.

        Args:
            config: TitanMACConfig with block sparse settings
            layer_idx: Index of this layer (0-based)

        Returns:
            True if this layer should use CMSBlockLinear for MLP
        """
        # Check if block-sparse MLP is enabled globally
        if not getattr(config, "mlp_use_block_sparse", False):
            return False

        # Check if 'mlp' is in the enabled components (T064)
        components = getattr(config, "mlp_block_sparse_components", ("mlp",))
        if "mlp" not in components:
            return False

        # Check if this layer is in the specified layers list (T063)
        sparse_layers = getattr(config, "mlp_block_sparse_layers", None)
        if sparse_layers is not None:
            # Only specified layers use block-sparse
            return layer_idx in sparse_layers

        # If no layer list specified, all layers use block-sparse
        return True

    @staticmethod
    def _get_layer_density(config: TitanMACConfig, layer_idx: int) -> float:
        """Get the block-sparse density for this layer.

        Supports per-layer density override via dict mapping.

        Args:
            config: TitanMACConfig with block sparse settings
            layer_idx: Index of this layer (0-based)

        Returns:
            Density value for this layer (float between 0.1 and 1.0)
        """
        density = getattr(config, "mlp_block_sparse_density", 0.5)

        # Check if density is a per-layer dict (T065)
        if isinstance(density, dict):
            # Return layer-specific density if available, else default to 0.5
            return density.get(layer_idx, 0.5)

        # Single float value for all layers
        return density
