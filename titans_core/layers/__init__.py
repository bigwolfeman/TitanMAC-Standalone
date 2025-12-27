"""Block-sparse layers for TitanMAC with CMS topology updates.

This module provides block-sparse linear layers that integrate with the
Continuum Memory System (CMS) for dynamic topology optimization.

Key components:
- CMSBlockLinear: Drop-in replacement for nn.Linear with block sparsity
- BlockELLConfig: Configuration for Block-ELL sparse format
- BlockELLTensor: Sparse tensor representation
- TopologyStats: Statistics for monitoring topology state
- TopologyDecisionResult: Results from topology update steps

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

from .block_ell import (
    BlockELLConfig,
    BlockELLTensor,
    create_block_ell_from_dense,
    block_ell_to_dense,
    validate_block_ell_topology,
    initialize_block_ell_topology,
)
from .block_sparse import (
    CMSBlockLinear,
    TopologyStats,
    TopologyDecisionResult,
)

__all__ = [
    # Block-ELL format
    "BlockELLConfig",
    "BlockELLTensor",
    "create_block_ell_from_dense",
    "block_ell_to_dense",
    "validate_block_ell_topology",
    "initialize_block_ell_topology",
    # Block-sparse layer
    "CMSBlockLinear",
    "TopologyStats",
    "TopologyDecisionResult",
]
