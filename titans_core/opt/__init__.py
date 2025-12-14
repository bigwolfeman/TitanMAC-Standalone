"""Nested optimizer systems for TitanMAC.

Implements Nested Learning (NeurIPS 2025):
- DeepNestedOptimizer: Complete nested learning optimizer
- ContinuumOptimizer: Legacy optimizer with controller-modulated LRs
- DMGDOptimizer: Deep Momentum Gradient Descent
- CMS: Continuum Memory System for multi-frequency updates
"""

from .continuum_optimizer import ContinuumOptimizer
from .nested_controller import NestedController
from .param_groups import group_titans_params, infer_param_depth
from .deep_nested_optimizer import DeepNestedOptimizer, L2RegressionMomentum, ContinuumMemoryState
from .dmgd import DMGDOptimizer, MomentumMLP
from .cms import ContinuumMemorySystem
from .meta_trainer import UnrolledMetaTrainer, SimplifiedMetaTrainer, create_meta_trainer

__all__ = [
    # New unified optimizer (recommended)
    "DeepNestedOptimizer",
    "L2RegressionMomentum",
    "ContinuumMemoryState",
    # Meta-learning
    "UnrolledMetaTrainer",
    "SimplifiedMetaTrainer",
    "create_meta_trainer",
    # Legacy components
    "ContinuumOptimizer",
    "NestedController",
    "DMGDOptimizer",
    "MomentumMLP",
    "ContinuumMemorySystem",
    # Utilities
    "group_titans_params",
    "infer_param_depth",
]
