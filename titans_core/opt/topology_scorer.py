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
        """
        # EMA formula: new_ema = alpha * grad_norms + (1 - alpha) * current_ema
        return self.ema_alpha * grad_norms + (1.0 - self.ema_alpha) * current_ema

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
        """
        # Outer product: [R] x [C] -> [R, C]
        # error_norms[:, None] is [R, 1], activation_norms[None, :] is [1, C]
        return torch.outer(error_norms, activation_norms)

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
        4. Apply swap_threshold: only swap if candidate is 1.5x better
        5. Select top-K scoring positions

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
        """
        R, K = current_scores.shape
        C = candidate_scores.shape[1]
        device = current_scores.device
        dtype = current_scores.dtype

        pruned_positions: List[Tuple[int, int]] = []
        grown_columns: List[int] = []

        # Create mask for active columns per row [R, C]
        active_mask = torch.zeros((R, C), dtype=torch.bool, device=device)
        active_mask.scatter_(1, col_indices, True)

        # Apply swap threshold: boost current block scores by swap_threshold factor
        # This makes it harder for candidates to beat them (candidate must be 1.5x better)
        boosted_current = current_scores * self.swap_threshold

        # Build combined score matrix [R, C]:
        # - Active positions: use boosted current_scores
        # - Inactive positions: use candidate_scores
        combined_scores = torch.full((R, C), float("-inf"), device=device, dtype=dtype)
        combined_scores.scatter_(1, col_indices, boosted_current)
        combined_scores = torch.where(active_mask, combined_scores, candidate_scores)

        # Epsilon-greedy exploration: with probability epsilon, add random noise
        if self.exploration_epsilon > 0:
            # Generate random mask for which positions get exploration noise
            explore_mask = (
                torch.rand((R, C), device=device, generator=generator) < self.exploration_epsilon
            )

            # Add large random noise to selected positions to potentially force selection
            # The noise should be large enough to override normal scoring
            max_score = (
                combined_scores[combined_scores != float("-inf")].max().item()
                if (combined_scores != float("-inf")).any()
                else 1.0
            )
            noise = torch.rand((R, C), device=device, generator=generator) * max_score * 2

            # Only apply noise to inactive positions (exploring new blocks, not keeping bad old ones)
            explore_mask = explore_mask & ~active_mask
            combined_scores = torch.where(explore_mask, noise, combined_scores)

        # Select top-K per row
        _, new_col_indices = torch.topk(combined_scores, K, dim=1, sorted=True)
        new_col_indices = new_col_indices.sort(dim=1).values  # Keep sorted for consistency

        # Determine which positions were pruned and which were grown
        # Convert to sets for comparison
        for r in range(R):
            old_cols = set(col_indices[r].tolist())
            new_cols = set(new_col_indices[r].tolist())

            # Pruned: was in old, not in new
            for old_slot, old_col in enumerate(col_indices[r].tolist()):
                if old_col not in new_cols:
                    pruned_positions.append((r, old_slot))

            # Grown: in new, not in old
            for new_col in new_cols:
                if new_col not in old_cols:
                    grown_columns.append(new_col)

        return new_col_indices, pruned_positions, grown_columns

    def compute_column_entropy(self, col_indices: Tensor) -> float:
        """Compute normalized entropy of column usage across all rows.

        High entropy means columns are used uniformly (good diversity).
        Low entropy means some columns are overused (poor diversity).

        Args:
            col_indices: Column indices [R, K]

        Returns:
            Normalized entropy in [0, 1]
        """
        # Count usage frequency of each column
        # col_indices is [R, K], we want to count how often each column index appears
        flat_indices = col_indices.flatten()  # [R * K]

        # Use bincount to count occurrences, with minlength=C
        counts = torch.bincount(flat_indices.to(torch.int64), minlength=self.C)  # [C]

        # Convert to probabilities (exclude zero counts from entropy calculation)
        total = counts.sum().float()
        if total == 0:
            return 0.0

        # Filter out zero counts to avoid log(0)
        nonzero_mask = counts > 0
        nonzero_counts = counts[nonzero_mask].float()

        # Compute probabilities
        probs = nonzero_counts / total

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs))

        # Normalize by max entropy (log(C)) to get value in [0, 1]
        if self.C <= 1:
            return 1.0  # Edge case: only one column means max entropy

        max_entropy = torch.log(torch.tensor(float(self.C), device=col_indices.device))
        normalized_entropy = (entropy / max_entropy).item()

        return normalized_entropy

    def should_swap(
        self,
        current_score: float,
        candidate_score: float,
        block_age: int,
    ) -> bool:
        """Determine if a swap should occur based on scores and age.

        A swap occurs if:
        - candidate_score > current_score * swap_threshold

        Note: block_age parameter is reserved for future use (age-based protection)
        but is currently ignored per task specification.

        Args:
            current_score: Score of current block
            candidate_score: Score of candidate position
            block_age: Age of current block in topology steps (reserved for future)

        Returns:
            True if swap should occur
        """
        # Simple threshold check: candidate must be swap_threshold times better
        return candidate_score > current_score * self.swap_threshold


def compute_gradient_frobenius_norms(grad: Tensor) -> Tensor:
    """Compute Frobenius norm of gradients per block.

    Frobenius norm = sqrt(sum of squared elements) per block.

    Args:
        grad: Gradient tensor [R, K, B, B]

    Returns:
        Frobenius norms [R, K]
    """
    # Frobenius norm: sqrt(sum of squared elements) per block
    # grad shape is [R, K, B, B], we want output [R, K]
    # Sum over the last two dimensions (B, B), then sqrt
    return torch.sqrt(torch.sum(grad * grad, dim=(-2, -1)))


def initialize_scores(
    R: int,
    C: int,
    K: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initialize scoring tensors for a new layer.

    Args:
        R: Number of output block-rows
        C: Number of input block-columns
        K: Active blocks per row
        device: Target device for tensors
        dtype: Data type for float tensors (block_age is always int32)

    Returns:
        Tuple of:
        - block_score_ema: [R, K] initialized to zeros
        - activation_norm_acc: [C] initialized to zeros
        - error_norm_acc: [R] initialized to zeros
        - block_age: [R, K] initialized to zeros (int32)
    """
    # Use defaults if not specified
    if dtype is None:
        dtype = torch.float32

    block_score_ema = torch.zeros(R, K, device=device, dtype=dtype)
    activation_norm_acc = torch.zeros(C, device=device, dtype=dtype)
    error_norm_acc = torch.zeros(R, device=device, dtype=dtype)
    block_age = torch.zeros(R, K, device=device, dtype=torch.int32)

    return block_score_ema, activation_norm_acc, error_norm_acc, block_age
