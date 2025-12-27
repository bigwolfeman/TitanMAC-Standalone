"""Topology scoring utilities for CMS block-sparse layers.

TopologyScorer manages the gradient-based importance metrics used to make
topology decisions in CMSBlockLinear layers. It handles:
- Gradient EMA accumulation for existing blocks
- Candidate scoring using activation x error product
- Epsilon-greedy exploration for topology diversity
- Top-K selection per row

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class BlockScore:
    """Per-block importance score with metadata.

    Attributes:
        row_idx: Output block-row index [0, R)
        slot_idx: Slot within row [0, K)
        col_idx: Input block-column index [0, C)
        gradient_ema: EMA of gradient Frobenius norm
        age: Steps since block creation
    """

    row_idx: int
    slot_idx: int
    col_idx: int
    gradient_ema: float
    age: int


@dataclass
class CandidateScore:
    """Score for a potential new block position.

    Attributes:
        row_idx: Output block-row index
        col_idx: Candidate input block-column
        score: activation_norm[col] * error_norm[row]
        is_active: Whether this position is currently active
    """

    row_idx: int
    col_idx: int
    score: float
    is_active: bool


class TopologyScorer:
    """Manages topology scoring and selection for block-sparse layers.

    This class implements the scoring logic for CMS Level 2 topology updates.
    It is designed to work with CMSBlockLinear but can be used standalone
    for testing and debugging.

    Args:
        R: Number of output block-rows
        C: Number of input block-columns
        K: Active blocks per row
        ema_alpha: Momentum for gradient EMA (default 0.95)
        exploration_epsilon: Random swap probability (default 0.05)
        swap_threshold: Required improvement ratio for swap (default 1.5)

    Example:
        >>> scorer = TopologyScorer(R=160, C=40, K=20)
        >>> grad_norms = torch.randn(160, 20).abs()  # [R, K]
        >>> scorer.update_gradient_ema(grad_norms)
    """

    def __init__(
        self,
        R: int,
        C: int,
        K: int,
        ema_alpha: float = 0.95,
        exploration_epsilon: float = 0.05,
        swap_threshold: float = 1.5,
    ) -> None:
        """Initialize topology scorer."""
        if K > C:
            raise ValueError(f"K ({K}) cannot exceed C ({C})")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha ({ema_alpha}) must be in [0, 1]")
        if not (0.0 <= exploration_epsilon <= 0.5):
            raise ValueError(f"exploration_epsilon ({exploration_epsilon}) must be in [0, 0.5]")

        self.R = R
        self.C = C
        self.K = K
        self.ema_alpha = ema_alpha
        self.exploration_epsilon = exploration_epsilon
        self.swap_threshold = swap_threshold

    def update_gradient_ema(
        self,
        grad_norms: Tensor,
        current_ema: Tensor,
    ) -> Tensor:
        """Update gradient EMA with new gradient norms.

        Implements: ema = alpha * new + (1 - alpha) * old

        Args:
            grad_norms: New gradient Frobenius norms [R, K]
            current_ema: Current EMA values [R, K]

        Returns:
            Updated EMA tensor [R, K]

        Raises:
            NotImplementedError: Skeleton - not yet implemented
        """
        raise NotImplementedError("update_gradient_ema not yet implemented")

    def compute_candidate_scores(
        self,
        activation_norms: Tensor,
        error_norms: Tensor,
    ) -> Tensor:
        """Compute candidate scores for all possible block positions.

        Score for position (r, c) = error_norm[r] * activation_norm[c]
        High error at output r AND high activation at input c suggests
        connecting r to c would help learning.

        Args:
            activation_norms: Input activation L2 norms [C]
            error_norms: Output gradient L2 norms [R]

        Returns:
            Candidate score matrix [R, C]

        Raises:
            NotImplementedError: Skeleton - not yet implemented
        """
        raise NotImplementedError("compute_candidate_scores not yet implemented")

    def select_top_k(
        self,
        current_scores: Tensor,
        candidate_scores: Tensor,
        col_indices: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, List[Tuple[int, int]], List[int]]:
        """Select top-K blocks per row with epsilon-greedy exploration.

        For each row:
        1. Get scores for current blocks from current_scores
        2. Get scores for candidate positions from candidate_scores
        3. With probability epsilon, add random noise for exploration
        4. Select top-K scoring positions

        Args:
            current_scores: Gradient EMA for current blocks [R, K]
            candidate_scores: Outer product scores [R, C]
            col_indices: Current column indices [R, K]
            generator: RNG for deterministic exploration

        Returns:
            Tuple of:
            - new_col_indices: Updated column indices [R, K]
            - pruned_positions: List of (row, slot) that were pruned
            - grown_columns: List of new column indices added

        Raises:
            NotImplementedError: Skeleton - not yet implemented
        """
        raise NotImplementedError("select_top_k not yet implemented")

    def compute_column_entropy(self, col_indices: Tensor) -> float:
        """Compute normalized entropy of column usage across all rows.

        High entropy means columns are used uniformly (good diversity).
        Low entropy means some columns are overused (poor diversity).

        Args:
            col_indices: Column indices [R, K]

        Returns:
            Normalized entropy in [0, 1]

        Raises:
            NotImplementedError: Skeleton - not yet implemented
        """
        raise NotImplementedError("compute_column_entropy not yet implemented")

    def should_swap(
        self,
        current_score: float,
        candidate_score: float,
        block_age: int,
    ) -> bool:
        """Determine if a swap should occur based on scores and age.

        A swap occurs if:
        - candidate_score > current_score * swap_threshold, OR
        - block_age > age_threshold (stale blocks get replaced)

        Args:
            current_score: Score of current block
            candidate_score: Score of candidate position
            block_age: Age of current block in topology steps

        Returns:
            True if swap should occur

        Raises:
            NotImplementedError: Skeleton - not yet implemented
        """
        raise NotImplementedError("should_swap not yet implemented")


def compute_gradient_frobenius_norms(grad: Tensor) -> Tensor:
    """Compute Frobenius norm of gradients per block.

    Args:
        grad: Gradient tensor [R, K, B, B]

    Returns:
        Frobenius norms [R, K]

    Raises:
        NotImplementedError: Skeleton - not yet implemented
    """
    raise NotImplementedError("compute_gradient_frobenius_norms not yet implemented")


def initialize_scores(
    R: int,
    C: int,
    K: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initialize scoring tensors for a new layer.

    Returns:
        Tuple of:
        - block_score_ema: [R, K] initialized to zeros
        - activation_norm_acc: [C] initialized to zeros
        - error_norm_acc: [R] initialized to zeros
        - block_age: [R, K] initialized to zeros (int32)

    Raises:
        NotImplementedError: Skeleton - not yet implemented
    """
    raise NotImplementedError("initialize_scores not yet implemented")
