"""TitanMAC: Memory-Augmented Transformer with Neural Long-Term Memory.

This package implements the TitanMAC architecture based on:
- Titans paper (arxiv 2501.00663): Neural Long-Term Memory
- Nested Learning paper (NeurIPS 2025): DMGD and CMS

Architecture features:
- Windowed attention with persistent tokens (O(T*w) memory)
- Neural Long-Term Memory with gradient-based surprise updates
- MAC/MAG/MAL architecture variants
- Deep Momentum Gradient Descent (DMGD)
- Continuum Memory System (CMS) for multi-frequency updates

Version: 0.2.0
License: MIT
"""

__version__ = "0.2.0"

# Import config
from .config import TitanMACConfig, TITANS_VARIANTS

# Import normalization and blocks
from .blocks.norms import RMSNorm
from .blocks.mlp import MLPBlock
from .blocks.titan_block import TitanBlock

# Import attention
from .attn.windowed_attention import WindowedAttention, WindowConfig

# Import memory
from .memory.memory_bank import MemoryBank
from .memory.neural_memory import NeuralMemory, ForgetGate, DecayGate

# Import models
from .models.titanmac import TitanMAC

# Import optimizer components
from .opt.continuum_optimizer import ContinuumOptimizer
from .opt.nested_controller import NestedController
from .opt.param_groups import group_titans_params, infer_param_depth
from .opt.dmgd import DMGDOptimizer, MomentumMLP
from .opt.cms import ContinuumMemorySystem

__all__ = [
    "__version__",
    # Config
    "TitanMACConfig",
    "TITANS_VARIANTS",
    # Blocks
    "RMSNorm",
    "MLPBlock",
    "TitanBlock",
    # Attention
    "WindowedAttention",
    "WindowConfig",
    # Memory
    "MemoryBank",
    "NeuralMemory",
    "ForgetGate",
    "DecayGate",
    # Models
    "TitanMAC",
    # Optimizers
    "ContinuumOptimizer",
    "NestedController",
    "group_titans_params",
    "infer_param_depth",
    "DMGDOptimizer",
    "MomentumMLP",
    "ContinuumMemorySystem",
]
