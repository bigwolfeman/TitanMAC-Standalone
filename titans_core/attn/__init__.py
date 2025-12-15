"""Attention mechanisms for TitanMAC.

Includes multi-head attention, retention mechanisms, and hybrid attention patterns.

Two implementations available:
- WindowedAttention: O(T²) masked attention (legacy, uses SDPA)
- BlockSparseAttention: O(T*w) true sparse attention (paper-faithful)
"""

from .windowed_attention import WindowedAttention, WindowConfig
from .block_sparse_attention import BlockSparseAttention, create_block_sparse_attention

__all__ = [
    "WindowedAttention",
    "WindowConfig",
    "BlockSparseAttention",
    "create_block_sparse_attention",
]
