"""
Parameter grouping utilities for ContinuumOptimizer.

Separates model parameters into:
- Core params: embeddings, attention, MLP, norms, LM head
- Memory params: MemoryBank components (if enabled)

Task: T046
"""

import re
from typing import Tuple, List
import torch.nn as nn


def group_titans_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Group model parameters into core and memory groups.

    Args:
        model: TitanMAC model instance

    Returns:
        Tuple of (core_params, memory_params) lists

    Core params include:
        - Token and position embeddings
        - Attention projections (Q, K, V, O)
        - MLP weights
        - Layer norms
        - LM head

    Memory params include:
        - MemoryBank MLP weights
        - Fast-weight matrices (U, V)
        - Memory gating weights

    Example:
        >>> model = TitanMAC(config)
        >>> core, memory = group_titans_params(model)
        >>> print(f"Core: {len(core)}, Memory: {len(memory)}")
    """
    core_params = []
    memory_params = []

    # Check if model has memory bank
    has_memory = hasattr(model, "memory_bank") and model.memory_bank is not None

    for name, param in model.named_parameters():
        # Memory bank parameters
        if has_memory and "memory_bank" in name:
            memory_params.append(param)
        else:
            # All other parameters go to core
            core_params.append(param)

    return core_params, memory_params


def infer_param_depth(name: str, n_layers: int) -> float:
    """
    Infer relative depth of parameter from name.

    Args:
        name: Parameter name (e.g., "blocks.5.mlp.fc1.weight")
        n_layers: Total number of layers

    Returns:
        Normalized depth in [0, 1] where 0=embedding, 1=output

    Example:
        >>> infer_param_depth("blocks.8.mlp.weight", n_layers=16)
        0.5  # Layer 8 is middle of 16-layer model

    Depth mapping:
        - Embeddings: 0.0
        - Layer i: (i + 1) / n_layers
        - Final norm: 1.0
        - LM head: 1.0
    """
    # Embedding layers: depth 0
    if "embed" in name.lower():
        return 0.0

    # Output layers: depth 1
    if "lm_head" in name.lower() or (name == "norm.weight" or name == "norm.bias"):
        return 1.0

    # Try to extract layer index from name
    # Patterns: "layers.5.", "blocks.5.", "layer.5."
    layer_patterns = [
        r"layers\.(\d+)\.",
        r"blocks\.(\d+)\.",
        r"layer\.(\d+)\.",
    ]

    for pattern in layer_patterns:
        match = re.search(pattern, name)
        if match:
            layer_idx = int(match.group(1))
            # Normalize to [0, 1] range
            # Layer 0 -> small positive depth
            # Layer n_layers-1 -> depth near 1
            depth = (layer_idx + 1) / n_layers
            return depth

    # Default: middle depth for unknown params
    return 0.5
