"""MLP block for Titan-MAC."""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Simple 2-layer MLP with GELU activation.

    Supports optional block-sparse linear layers via CMSBlockLinear when
    `use_block_sparse=True`. Block-sparse layers provide memory efficiency
    and anti-forgetting capabilities through dynamic topology updates.

    Args:
        d_model: Model dimension (input/output)
        d_ff: Feed-forward dimension (hidden layer)
        dropout: Dropout probability (default: 0.0)
        use_block_sparse: If True, use CMSBlockLinear instead of nn.Linear
        block_sparse_tile_size: Tile size for block-sparse (8, 16, or 32)
        block_sparse_density: Fraction of active blocks per row (0.1 to 1.0)

    Shape:
        Input: [B, T, d_model]
        Output: [B, T, d_model]

    Example:
        >>> mlp = MLPBlock(d_model=128, d_ff=512)
        >>> x = torch.randn(2, 64, 128)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([2, 64, 128])

        >>> # With block-sparse (dimensions must be divisible by tile_size)
        >>> mlp_sparse = MLPBlock(d_model=128, d_ff=512, use_block_sparse=True)
        >>> y_sparse = mlp_sparse(x)
        >>> y_sparse.shape
        torch.Size([2, 64, 128])
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        use_block_sparse: bool = False,
        block_sparse_tile_size: int = 16,
        block_sparse_density: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.use_block_sparse = use_block_sparse
        self.block_sparse_tile_size = block_sparse_tile_size
        self.block_sparse_density = block_sparse_density

        # Two-layer MLP
        if use_block_sparse:
            # Import CMSBlockLinear - lazy import to avoid circular dependencies
            from titans_core.layers.block_sparse import CMSBlockLinear

            # Check dimension constraints for block-sparse
            tile_size = block_sparse_tile_size
            fc1_compatible = (d_model % tile_size == 0) and (d_ff % tile_size == 0)
            fc2_compatible = (d_ff % tile_size == 0) and (d_model % tile_size == 0)

            if fc1_compatible:
                # Up projection: d_model → d_ff (block-sparse)
                self.fc1 = CMSBlockLinear(
                    in_features=d_model,
                    out_features=d_ff,
                    tile_size=tile_size,
                    density=block_sparse_density,
                    bias=True,
                )
            else:
                # Fall back to dense if dimensions not compatible
                import warnings

                warnings.warn(
                    f"MLPBlock: fc1 dimensions ({d_model} -> {d_ff}) not divisible by "
                    f"tile_size={tile_size}, falling back to dense nn.Linear"
                )
                self.fc1 = nn.Linear(d_model, d_ff)

            if fc2_compatible:
                # Down projection: d_ff → d_model (block-sparse)
                self.fc2 = CMSBlockLinear(
                    in_features=d_ff,
                    out_features=d_model,
                    tile_size=tile_size,
                    density=block_sparse_density,
                    bias=True,
                )
            else:
                # Fall back to dense if dimensions not compatible
                import warnings

                warnings.warn(
                    f"MLPBlock: fc2 dimensions ({d_ff} -> {d_model}) not divisible by "
                    f"tile_size={tile_size}, falling back to dense nn.Linear"
                )
                self.fc2 = nn.Linear(d_ff, d_model)
        else:
            # Standard dense layers
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
