"""Attention mechanisms for TitanMAC.

Includes multi-head attention, retention mechanisms, and hybrid attention patterns.
"""

from .windowed_attention import WindowedAttention, WindowConfig

__all__ = ["WindowedAttention", "WindowConfig"]
