"""
Triton kernels for TitanMAC optimization.

These kernels replace PyTorch operations with fused Triton implementations
for better performance, especially for neural memory operations.
"""

from .memory_mlp import (
    memory_mlp_backward_update,
    MemoryMLPBackwardUpdate,
)

__all__ = [
    'memory_mlp_backward_update',
    'MemoryMLPBackwardUpdate',
]
