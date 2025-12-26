"""
API Contract: CMSBlockLinear

This file defines the interface contract for the block-sparse linear layer.
Implementation must satisfy all method signatures and docstring contracts.

Date: 2025-12-25
Branch: 001-cms-block-sparse
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class TopologyStats:
    """Statistics about current topology state."""

    density: float  # Actual density (K/C)
    avg_block_score: float  # Mean gradient EMA across blocks
    avg_block_age: float  # Mean age in topology steps
    column_entropy: float  # Entropy of column usage (0-1 normalized)
    num_blocks: int  # Total active blocks (R × K)


@dataclass
class TopologyDecisionResult:
    """Result of a topology step."""

    num_swaps: int  # Blocks swapped this step
    swap_rate: float  # num_swaps / total_blocks
    pruned_positions: list  # List of (row, slot) pruned
    grown_columns: list  # List of new column indices grown


class CMSBlockLinear:
    """
    Block-sparse linear layer with dynamic topology via CMS Level 2 updates.

    Drop-in replacement for nn.Linear with:
    - Block-ELL sparse weight storage
    - Gradient-based importance scoring
    - Periodic topology updates (prune/grow blocks)

    Args:
        in_features: Input dimension (must be divisible by tile_size)
        out_features: Output dimension (must be divisible by tile_size)
        tile_size: Block size (default 16 for WMMA compatibility)
        density: Fraction of active blocks per row (0.1 to 1.0)
        bias: Include bias term (default True)

    Raises:
        ValueError: If dimensions not divisible by tile_size
        ValueError: If density not in [0.1, 1.0]

    Example:
        >>> layer = CMSBlockLinear(640, 2560, tile_size=16, density=0.5)
        >>> x = torch.randn(32, 640)
        >>> y = layer(x)  # [32, 2560]
    """

    # === Core Properties ===

    in_features: int
    out_features: int
    tile_size: int  # B
    density: float
    R: int  # out_features // tile_size
    C: int  # in_features // tile_size
    K: int  # int(C * density)

    # === Tensors ===

    values: Tensor  # [R, K, B, B] - learnable weights
    col_indices: Tensor  # [R, K] - topology (int32)
    bias: Optional[Tensor]  # [out_features] - learnable bias

    # === Scoring State ===

    block_score_ema: Tensor  # [R, K] - gradient importance
    activation_norm_acc: Tensor  # [C] - input activation norms
    error_norm_acc: Tensor  # [R] - output error norms
    block_age: Tensor  # [R, K] - steps since creation

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_size: int = 16,
        density: float = 0.5,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize block-sparse linear layer."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: block-sparse matrix multiplication.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor with same batch/seq dims, out_features last

        Contract:
            - Output shape matches nn.Linear contract
            - Numerically equivalent to dense @ sparse_mask at initialization
            - Supports 2D and 3D input tensors
        """
        ...

    def accumulate_scores(self) -> None:
        """
        Accumulate gradient statistics for importance scoring.

        Call after backward() each step. Updates:
        - block_score_ema: EMA of gradient Frobenius norms
        - activation_norm_acc: Accumulated input norms (requires hook)
        - error_norm_acc: Accumulated output error norms (requires hook)

        Contract:
            - Safe to call even if values.grad is None (no-op)
            - Accumulates into existing EMA (doesn't reset)
        """
        ...

    def score_step(self) -> None:
        """
        Level 1 update: normalize accumulators and increment ages.

        Call every ~10 training steps. Actions:
        - Normalize activation_norm_acc by step count
        - Normalize error_norm_acc by step count
        - Increment block_age for all active blocks
        - Reset step counter

        Contract:
            - Does NOT reset block_score_ema (kept for Level 2)
            - Does NOT change topology
        """
        ...

    def topology_step(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> TopologyDecisionResult:
        """
        Level 2 update: make topology decisions (prune/grow blocks).

        Call every ~100 training steps. Actions:
        1. Score existing blocks by gradient EMA
        2. Score candidates by activation × error product
        3. Apply epsilon-greedy exploration
        4. Select top-K per row
        5. Swap low-scoring blocks for high-scoring candidates
        6. Initialize new block weights
        7. Reset all accumulators

        Args:
            generator: RNG for DDP determinism (use same seed on all ranks)

        Returns:
            TopologyDecisionResult with swap statistics

        Contract:
            - Maintains exactly K active blocks per row
            - New blocks initialized with Kaiming ×0.1 scale
            - Resets block_age to 0 for new blocks
            - All accumulators reset after decision
        """
        ...

    def get_topology_stats(self) -> TopologyStats:
        """
        Get current topology statistics for logging.

        Returns:
            TopologyStats with density, scores, ages, entropy

        Contract:
            - Read-only (doesn't modify state)
            - Safe to call at any time
        """
        ...

    def get_density(self) -> float:
        """
        Get actual current density.

        Returns:
            K / C (should match configured density)
        """
        ...

    def state_dict(self) -> Dict[str, Any]:
        """
        Return state for checkpointing.

        Includes:
            - values: Weight parameters
            - col_indices: Topology
            - bias: Bias parameters (if present)
            - block_score_ema: Scoring state
            - block_age: Block ages
            - accumulators: Accumulated norms
        """
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint.

        Restores full layer state including topology and scoring.
        """
        ...

    def to_dense(self) -> Tensor:
        """
        Convert current topology to dense weight matrix.

        Returns:
            Dense weight tensor [out_features, in_features]

        Use for:
            - Debugging / visualization
            - Comparison with nn.Linear
            - Export to frameworks without sparse support
        """
        ...

    @classmethod
    def from_dense(
        cls,
        dense_layer: torch.nn.Linear,
        tile_size: int = 16,
        density: float = 0.5,
    ) -> "CMSBlockLinear":
        """
        Create sparse layer from existing dense layer.

        Initializes topology by magnitude-based pruning of dense weights.

        Args:
            dense_layer: Source nn.Linear layer
            tile_size: Block size for sparse format
            density: Target density

        Returns:
            CMSBlockLinear initialized from dense weights
        """
        ...


# === Integration with DeepNestedOptimizer ===

class TopologyAwareOptimizer:
    """
    Protocol for optimizers that manage block-sparse topology.

    DeepNestedOptimizer should implement these methods when
    block-sparse layers are detected in the model.
    """

    def discover_block_sparse_layers(self, model: torch.nn.Module) -> list:
        """
        Find all CMSBlockLinear layers in model.

        Called during optimizer initialization.
        """
        ...

    def topology_schedule_step(self) -> int:
        """
        Called at Level 2 frequency to trigger topology updates.

        Returns:
            Total blocks swapped across all layers
        """
        ...

    def sync_topology_scores(self) -> None:
        """
        Synchronize scores across DDP ranks before topology step.

        Uses all-reduce to average:
        - block_score_ema
        - activation_norm_acc
        - error_norm_acc
        """
        ...
