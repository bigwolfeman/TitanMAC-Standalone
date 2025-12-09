"""
Windowed attention with persistent tokens for Titan-MAC.

This module provides windowed attention with SDPA (Scaled Dot-Product Attention)
and persistent tokens that attend bidirectionally.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WindowConfig:
    """
    Configuration for windowed attention.

    Args:
        window_size: Window radius for local attention
        num_heads: Number of attention heads
        head_dim: Dimension per head
        n_persistent: Number of persistent (global) tokens
        dropout: Dropout probability
        causal: Apply causal masking (default: True)
    """
    window_size: int
    num_heads: int
    head_dim: int
    n_persistent: int
    dropout: float = 0.0
    causal: bool = True


def enable_sdpa_backends(
    mem_efficient: bool = True,
    math: bool = True,
    flash: bool = False
) -> None:
    """
    Configure PyTorch SDPA backends for compatibility.

    On Windows with CUDA 12.x:
    - Memory-efficient backend (xFormers): ENABLED - 20x faster than math backend
    - Math backend: ENABLED - fallback for unsupported cases
    - Flash Attention: DISABLED - not available on Windows

    Args:
        mem_efficient: Enable memory-efficient attention (xFormers) (default: True)
        math: Enable math fallback backend (default: True)
        flash: Enable Flash Attention (default: False, not supported on Windows)
    """
    # Set global SDPA backend flags
    torch.backends.cuda.enable_mem_efficient_sdp(mem_efficient)
    torch.backends.cuda.enable_math_sdp(math)
    torch.backends.cuda.enable_flash_sdp(flash)


@lru_cache(maxsize=32)
def build_band_mask(
    seq_len: int,
    window_size: int,
    num_persistent_tokens: int,
    device: str = 'cuda',
    causal: bool = True
) -> torch.Tensor:
    """
    Build band attention mask for window attention with persistent tokens.

    Attention pattern:
    1. Persistent tokens (first Np tokens):
       - Attend to all other tokens bidirectionally
       - All tokens attend to persistent tokens
    2. Sequence tokens (remaining T tokens):
       - Attend within local window: |i - j| <= window_size
       - Causal masking: only attend to past (i >= j) if causal=True

    Args:
        seq_len: Total sequence length including persistent tokens (Np + T)
        window_size: Window radius for local attention (w in |i-j| <= w)
        num_persistent_tokens: Number of persistent tokens (Np)
        device: Device for mask tensor ('cuda' or 'cpu')
        causal: Apply causal masking (attend only to past) (default: True)

    Returns:
        torch.Tensor: Boolean mask of shape (seq_len, seq_len)
            - True: allow attention
            - False: mask out (prevent attention)

    Example:
        >>> mask = build_band_mask(4096, 512, 16, device='cuda')
        >>> print(mask.shape)
        torch.Size([4096, 4096])
    """
    if num_persistent_tokens >= seq_len:
        raise ValueError(
            f"num_persistent_tokens ({num_persistent_tokens}) must be < seq_len ({seq_len})"
        )

    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")

    # Create position indices
    positions = torch.arange(seq_len, device=device)
    row_pos = positions.unsqueeze(1)  # (seq_len, 1)
    col_pos = positions.unsqueeze(0)  # (1, seq_len)

    # Initialize mask (True = allow attention)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # === Persistent tokens (first Np tokens) ===
    # Persistent tokens attend to each other (bidirectional within persistent region)
    mask[:num_persistent_tokens, :num_persistent_tokens] = True

    # All tokens attend to persistent tokens (persistent = global context)
    mask[:, :num_persistent_tokens] = True

    # CRITICAL: Persistent tokens attend CAUSALLY to sequence tokens
    # Without this, persistent tokens see future and leak info through later layers
    if causal:
        # Persistent token at position p can only attend to sequence positions <= p
        # Since persistent positions are 0..Np-1 and sequence starts at Np,
        # persistent tokens can attend to sequence positions Np..Np+t where t corresponds
        # to the "virtual" position. But persistent tokens represent "global" context,
        # so they should only see past sequence tokens up to the current query position.
        #
        # For training with packed sequences, the safest approach is:
        # persistent tokens do NOT attend to sequence tokens at all (only to each other)
        # This prevents any future leakage while preserving global context.
        pass  # Persistent tokens already set to attend only to persistent region above

    # === Sequence tokens (remaining T tokens) ===
    # Band mask: |i - j| <= window_size
    distance = torch.abs(row_pos - col_pos)
    band_mask = distance <= window_size

    # Apply band mask to sequence tokens (excluding persistent region)
    seq_start = num_persistent_tokens
    mask[seq_start:, seq_start:] = band_mask[seq_start:, seq_start:]

    # === Causal masking ===
    if causal:
        # Only attend to past: i >= j (or equivalently, row >= col)
        causal_mask = row_pos >= col_pos

        # Apply causal constraint to sequence tokens
        # (persistent tokens remain bidirectional)
        mask[seq_start:, seq_start:] = mask[seq_start:, seq_start:] & causal_mask[seq_start:, seq_start:]

    return mask


def convert_mask_to_sdpa_format(
    mask: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert boolean mask to SDPA additive mask format.

    SDPA expects additive masks:
    - 0.0: allow attention
    - -inf: mask out (prevent attention)

    Args:
        mask: Boolean mask (True = allow, False = mask out)
        dtype: Output dtype (default: float32)

    Returns:
        torch.Tensor: Additive mask for SDPA
    """
    # SDPA format: 0.0 for allowed, -inf for masked
    sdpa_mask = torch.zeros_like(mask, dtype=dtype)
    sdpa_mask = sdpa_mask.masked_fill(~mask, float('-inf'))
    return sdpa_mask


class WindowedAttention(nn.Module):
    """
    Windowed attention with persistent tokens.

    Implements windowed attention where:
    - Persistent tokens attend bidirectionally to all positions
    - Sequence tokens attend within a local window
    - Optional causal masking for autoregressive models

    Memory complexity: O(B * n_heads * T * (w + n_persistent))

    Args:
        config: WindowConfig or TitanMACConfig with attention parameters

    Shape:
        Input: [B, T, d_model]
        Output: [B, T, d_model]

    Example:
        >>> from titans_core.config import TitanMACConfig
        >>> config = TitanMACConfig(d_model=640, n_heads=10, window_size=512)
        >>> attn = WindowedAttention(config)
        >>> x = torch.randn(2, 512, 640)
        >>> out = attn(x)
    """

    def __init__(self, config):
        super().__init__()

        # Handle both WindowConfig and TitanMACConfig
        if hasattr(config, 'num_heads'):
            # WindowConfig
            self.d_model = config.num_heads * config.head_dim
            self.n_heads = config.num_heads
            self.d_head = config.head_dim
            self.window_size = config.window_size
            self.num_persistent = config.n_persistent
            self.causal = config.causal
            self.dropout_p = config.dropout
        else:
            # TitanMACConfig
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_head
            self.window_size = config.window_size
            self.num_persistent = config.n_persistent
            self.causal = config.causal
            self.dropout_p = config.attention_dropout

        # Validate dimensions
        assert self.d_model == self.n_heads * self.d_head, \
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * d_head ({self.d_head})"

        # Q/K/V projections
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # Cache for attention mask
        self._mask_cache = {}

    def _get_attention_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Get or create attention mask for given sequence length.

        Creates windowed+causal mask with bidirectional persistent token attention.

        Args:
            seq_len: Sequence length (including persistent tokens)
            device: Device for mask tensor

        Returns:
            Boolean mask [seq_len, seq_len] where True = allow attention
        """
        # Check cache
        cache_key = (seq_len, str(device))
        if cache_key not in self._mask_cache:
            # Use build_band_mask for proper persistent token handling
            mask = build_band_mask(
                seq_len=seq_len,
                window_size=self.window_size,
                num_persistent_tokens=self.num_persistent,
                device=str(device),
                causal=self.causal
            )

            self._mask_cache[cache_key] = mask

        return self._mask_cache[cache_key]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through windowed attention.

        Args:
            x: Input tensor [B, T, d_model]
            attn_mask: Optional custom attention mask [T, T]

        Returns:
            Output tensor [B, T, d_model]
        """
        B, T, D = x.shape

        # Q/K/V projection
        qkv = self.qkv_proj(x)  # [B, T, 3*d_model]

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, T, d_model]

        # Reshape for multi-head attention
        # [B, T, d_model] → [B, T, n_heads, d_head] → [B, n_heads, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Build attention mask
        if attn_mask is None:
            attn_mask = self._get_attention_mask(T, x.device)

        # Convert boolean mask to additive mask for SDPA
        if attn_mask.dtype == torch.bool:
            sdpa_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
            sdpa_mask.masked_fill_(~attn_mask, float('-inf'))
        else:
            sdpa_mask = attn_mask

        # Apply SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # We provide explicit mask
        )
        # attn_output: [B, n_heads, T, d_head]

        # Concat heads
        attn_output = attn_output.transpose(1, 2)  # [B, T, n_heads, d_head]
        attn_output = attn_output.reshape(B, T, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        return output
