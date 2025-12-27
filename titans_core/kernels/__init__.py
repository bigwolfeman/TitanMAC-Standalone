"""
Triton kernels for TitanMAC optimization.

These kernels replace PyTorch operations with fused Triton implementations
for better performance, especially for neural memory operations.

Kernel modules:
- memory_mlp: Fused neural memory MLP backward + update
- block_ell_forward: Block-ELL sparse forward pass
- block_ell_backward: Block-ELL sparse backward pass
"""

from .memory_mlp import (
    memory_mlp_backward_update,
    MemoryMLPBackwardUpdate,
)

from .block_ell_forward import (
    block_ell_forward,
    block_ell_forward_autograd,
    block_ell_forward_reference,
    TRITON_AVAILABLE,
)

from .block_ell_backward import (
    block_ell_backward,
    block_ell_backward_dx,
    block_ell_backward_dw,
    block_ell_backward_db,
    BlockELLFunction,
    block_ell_autograd,
)

__all__ = [
    # Memory MLP kernels
    'memory_mlp_backward_update',
    'MemoryMLPBackwardUpdate',
    # Block-ELL forward kernels
    'block_ell_forward',
    'block_ell_forward_autograd',
    'block_ell_forward_reference',
    'TRITON_AVAILABLE',
    # Block-ELL backward kernels
    'block_ell_backward',
    'block_ell_backward_dx',
    'block_ell_backward_dw',
    'block_ell_backward_db',
    'BlockELLFunction',
    'block_ell_autograd',
]
