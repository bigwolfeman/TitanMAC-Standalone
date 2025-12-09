"""Nested optimizer systems for TitanMAC.

Includes MAG optimizer, surprise computation, and memory update mechanisms.
"""

from .continuum_optimizer import ContinuumOptimizer
from .nested_controller import NestedController
from .param_groups import group_titans_params, infer_param_depth

__all__ = [
    "ContinuumOptimizer",
    "NestedController",
    "group_titans_params",
    "infer_param_depth",
]
