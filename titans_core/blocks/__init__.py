"""Core building blocks for TitanMAC architecture.

Includes MAC blocks, transformer-like layers, and component compositions.
"""

from .norms import RMSNorm
from .mlp import MLPBlock
from .titan_block import TitanBlock

__all__ = ["RMSNorm", "MLPBlock", "TitanBlock"]
