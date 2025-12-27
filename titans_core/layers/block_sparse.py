"""Block-sparse linear layer with dynamic CMS topology updates.

CMSBlockLinear is a drop-in replacement for nn.Linear that uses:
- Block-ELL sparse weight storage for memory efficiency
- Gradient-based importance scoring for topology decisions
- Periodic topology updates (prune/grow blocks) via CMS Level 2

Date: 2025-12-26
Branch: 001-cms-block-sparse
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .block_ell import BlockELLConfig
from titans_core.opt.topology_scorer import TopologyScorer, compute_gradient_frobenius_norms

# Check if Triton is available
try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@dataclass
class TopologyStats:
    """Statistics about current topology state.

    Attributes:
        density: Actual density (K/C)
        avg_block_score: Mean gradient EMA across blocks
        avg_block_age: Mean age in topology steps
        column_entropy: Entropy of column usage (0-1 normalized)
        num_blocks: Total active blocks (R * K)
    """

    density: float
    avg_block_score: float
    avg_block_age: float
    column_entropy: float
    num_blocks: int


@dataclass
class TopologyDecisionResult:
    """Result of a topology step.

    Attributes:
        num_swaps: Blocks swapped this step
        swap_rate: num_swaps / total_blocks
        pruned_positions: List of (row, slot) pruned
        grown_columns: List of new column indices grown
    """

    num_swaps: int
    swap_rate: float
    pruned_positions: List[Tuple[int, int]]
    grown_columns: List[int]


class CMSBlockLinear(nn.Module):
    """Block-sparse linear layer with dynamic topology via CMS Level 2 updates.

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
        score_ema_alpha: EMA momentum for gradient scores (default 0.95)
        device: Target device
        dtype: Parameter dtype

    Raises:
        ValueError: If dimensions not divisible by tile_size
        ValueError: If density not in [0.1, 1.0]

    Example:
        >>> layer = CMSBlockLinear(640, 2560, tile_size=16, density=0.5)
        >>> x = torch.randn(32, 640)
        >>> y = layer(x)  # [32, 2560]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_size: int = 16,
        density: float = 0.5,
        bias: bool = True,
        score_ema_alpha: float = 0.95,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize block-sparse linear layer."""
        super().__init__()

        # Validate inputs
        if in_features % tile_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tile_size ({tile_size})"
            )
        if out_features % tile_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tile_size ({tile_size})"
            )
        if not (0.1 <= density <= 1.0):
            raise ValueError(f"density ({density}) must be in [0.1, 1.0]")

        # Core dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.density = density
        self.score_ema_alpha = score_ema_alpha

        # Derived dimensions
        self.R = out_features // tile_size  # output block-rows
        self.C = in_features // tile_size  # input block-columns
        self.K = max(1, int(self.C * density))  # active blocks per row

        # Block-ELL config
        self._block_ell_config = BlockELLConfig(R=self.R, C=self.C, K=self.K, B=tile_size)

        # Weight parameters [R, K, B, B]
        self.values = nn.Parameter(
            torch.empty(self.R, self.K, tile_size, tile_size, device=device, dtype=dtype)
        )

        # Topology: column indices [R, K] - not a parameter (int32)
        self.register_buffer(
            "col_indices",
            torch.zeros(self.R, self.K, dtype=torch.int32, device=device),
        )

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # === Scoring State (buffers, not parameters) ===

        # Gradient importance EMA [R, K]
        self.register_buffer(
            "block_score_ema",
            torch.zeros(self.R, self.K, device=device, dtype=dtype or torch.float32),
        )

        # Input activation norms accumulated [C]
        self.register_buffer(
            "activation_norm_acc",
            torch.zeros(self.C, device=device, dtype=dtype or torch.float32),
        )

        # Output error norms accumulated [R]
        self.register_buffer(
            "error_norm_acc",
            torch.zeros(self.R, device=device, dtype=dtype or torch.float32),
        )

        # Block age (steps since creation) [R, K]
        self.register_buffer(
            "block_age",
            torch.zeros(self.R, self.K, dtype=torch.int32, device=device),
        )

        # Step counter for accumulator normalization
        self._acc_steps: int = 0

        # Topology history for monitoring (T077)
        # Stores (before_col_indices, after_col_indices) tuples
        self._topology_history: List[Tuple[Tensor, Tensor]] = []
        self._topology_history_max_size: int = 10  # Limit memory usage

        # Swap rate tracking (T074)
        self._swap_rate_history: List[float] = []
        self._swap_rate_history_max_size: int = 100  # Rolling window

        # Initialize weights and topology
        self._reset_parameters()
        self._initialize_topology()

        # Register hooks for activation and gradient capture (T029, T030)
        self.register_forward_hook(self._activation_hook)
        self.register_full_backward_hook(self._gradient_hook)

    def _reset_parameters(self) -> None:
        """Initialize weight parameters with Kaiming uniform."""
        # Scale factor for sparse initialization
        # Use fan_in based on actual connections, not full dense
        fan_in = self.K * self.tile_size
        gain = nn.init.calculate_gain("relu")
        std = gain / (fan_in**0.5)
        nn.init.normal_(self.values, mean=0.0, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _initialize_topology(self) -> None:
        """Initialize column indices with random topology."""
        # Random unique columns per row
        for r in range(self.R):
            perm = torch.randperm(self.C, device=self.col_indices.device)
            self.col_indices[r] = perm[: self.K].to(torch.int32)

    @property
    def _use_triton_kernel(self) -> bool:
        """T050: Detect whether to use Triton kernels for forward/backward.

        Triton kernels are used when:
        1. Triton is installed and available
        2. Tensors are on CUDA device
        3. Tile size is compatible (8, 16, 32, or 64)

        Returns:
            True if Triton kernels should be used
        """
        return TRITON_AVAILABLE and self.values.is_cuda and self.tile_size in (8, 16, 32, 64)

    def _activation_hook(
        self, module: nn.Module, input: Tuple[Tensor, ...], output: Tensor
    ) -> None:
        """T029: Capture input activation norms per block-column.

        Computes L2 norm of input activations for each block-column and accumulates
        into activation_norm_acc buffer. Only active during training.

        Args:
            module: The module (self)
            input: Tuple containing input tensor(s)
            output: Output tensor (not used)
        """
        if not self.training:
            return

        x = input[0]  # [batch, in_features] or [batch, seq, in_features]
        B = self.tile_size

        # Handle 2D vs 3D input
        if x.dim() == 2:
            # [batch, in_features] -> [batch, 1, in_features]
            x = x.unsqueeze(1)

        # x is now [batch, seq, in_features]
        batch_size, seq_len, _ = x.shape

        # Reshape to block view: [batch, seq, C, B]
        x_blocks = x.view(batch_size, seq_len, self.C, B)

        # Compute L2 norm per block-column: sqrt(sum of squared elements)
        # First flatten batch and seq: [batch * seq, C, B]
        x_flat = x_blocks.view(-1, self.C, B)

        # L2 norm per column: [C]
        # Sum over batch and feature dimensions, then sqrt
        col_norms = torch.sqrt(torch.sum(x_flat * x_flat, dim=(0, 2)))

        # Accumulate into buffer (detach to avoid autograd issues)
        with torch.no_grad():
            self.activation_norm_acc = self.activation_norm_acc + col_norms.detach()

    def _gradient_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[Tensor], ...],
        grad_output: Tuple[Tensor, ...],
    ) -> None:
        """T030: Capture output gradient norms per block-row.

        Computes L2 norm of output gradients for each block-row and accumulates
        into error_norm_acc buffer. Only active during training.

        Args:
            module: The module (self)
            grad_input: Gradients with respect to inputs (not used)
            grad_output: Tuple containing gradient tensor(s) with respect to output
        """
        if not self.training:
            return

        grad = grad_output[0]  # [batch, out_features] or [batch, seq, out_features]
        if grad is None:
            return

        B = self.tile_size

        # Handle 2D vs 3D input
        if grad.dim() == 2:
            # [batch, out_features] -> [batch, 1, out_features]
            grad = grad.unsqueeze(1)

        # grad is now [batch, seq, out_features]
        batch_size, seq_len, _ = grad.shape

        # Reshape to block view: [batch, seq, R, B]
        grad_blocks = grad.view(batch_size, seq_len, self.R, B)

        # Compute L2 norm per block-row: sqrt(sum of squared elements)
        # First flatten batch and seq: [batch * seq, R, B]
        grad_flat = grad_blocks.view(-1, self.R, B)

        # L2 norm per row: [R]
        # Sum over batch and feature dimensions, then sqrt
        row_norms = torch.sqrt(torch.sum(grad_flat * grad_flat, dim=(0, 2)))

        # Accumulate into buffer (detach to avoid autograd issues)
        with torch.no_grad():
            self.error_norm_acc = self.error_norm_acc + row_norms.detach()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: block-sparse matrix multiplication.

        T051: Dispatches to Triton kernel when available (CUDA + compatible tile size).
        T052: Falls back to PyTorch reference implementation when Triton unavailable.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor with same batch/seq dims, out_features last

        Contract:
            - Output shape matches nn.Linear contract
            - Numerically equivalent to dense @ sparse_mask at initialization
            - Supports 2D and 3D input tensors
        """
        # T051/T052: Dispatch to Triton kernel or PyTorch reference
        if self._use_triton_kernel:
            return self._forward_triton(x)
        else:
            return self._forward_reference(x)

    def _forward_triton(self, x: Tensor) -> Tensor:
        """Forward using Triton kernels with autograd support.

        Uses block_ell_forward_autograd which wraps the Triton kernel in
        torch.autograd.Function for proper gradient computation.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor
        """
        from titans_core.kernels.block_ell_forward import block_ell_forward_autograd

        return block_ell_forward_autograd(
            x=x,
            values=self.values,
            col_indices=self.col_indices,
            bias=self.bias,
            use_triton=True,
        )

    def _forward_reference(self, x: Tensor) -> Tensor:
        """Forward using PyTorch reference implementation.

        Used when Triton is unavailable (CPU, unsupported tile size).
        This implementation is correct but slower than the Triton version.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor
        """
        # Handle 2D vs 3D input
        if x.dim() == 2:
            # [batch, in_features] -> [batch, 1, in_features]
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        # x is now [batch, seq, in_features]
        batch_size, seq_len, _ = x.shape
        B = self.tile_size

        # Reshape input to block view: [batch, seq, C, B]
        x_blocks = x.view(batch_size, seq_len, self.C, B)

        # Initialize output: [batch, seq, R, B]
        output = torch.zeros(batch_size, seq_len, self.R, B, dtype=x.dtype, device=x.device)

        # Block-sparse matmul: for each output row, gather K input blocks,
        # multiply with weight tiles, and sum
        for r in range(self.R):
            # Get column indices for this row: [K]
            cols = self.col_indices[r]  # [K], int32

            # Gather input blocks at these columns: [batch, seq, K, B]
            # cols has shape [K], we need to gather along dim=2 (the C dimension)
            input_gathered = x_blocks[:, :, cols.long(), :]  # [batch, seq, K, B]

            # Weight tiles for this row: [K, B, B]
            weights = self.values[r]  # [K, B, B]

            # Matmul: input_gathered @ weights^T
            # [batch, seq, K, B] @ [K, B, B]^T -> [batch, seq, K, B]
            # We need einsum: (b, s, k, b_in) @ (k, b_out, b_in) -> (b, s, k, b_out)
            # Then sum over K dimension
            # Actually weights are [K, B, B] where first B is output, second B is input
            # So we do: input [b, s, k, B_in] @ weight [k, B_out, B_in]^T
            # = input [b, s, k, B_in] @ weight.transpose(-1, -2) [k, B_in, B_out]
            # = result [b, s, k, B_out]

            # Compute per-block matmul and sum
            # weights: [K, B_out, B_in] where B_out = B_in = B in our case
            # input_gathered: [batch, seq, K, B_in]
            # We want: sum over k of (input_gathered[b,s,k,:] @ weights[k,:,:].T)

            # Using einsum: bskI,kOI->bskO then sum over k
            block_outputs = torch.einsum("bski,koi->bsko", input_gathered, weights)
            # Sum over K blocks: [batch, seq, B]
            row_output = block_outputs.sum(dim=2)  # [batch, seq, B]

            output[:, :, r, :] = row_output

        # Reshape output: [batch, seq, R, B] -> [batch, seq, out_features]
        output = output.view(batch_size, seq_len, self.out_features)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        # Squeeze back if input was 2D
        if squeeze_output:
            output = output.squeeze(1)  # [batch, out_features]

        return output

    def accumulate_scores(self) -> None:
        """T031: Accumulate gradient statistics for importance scoring.

        Call after backward() each step. Updates:
        - block_score_ema: EMA of gradient Frobenius norms
        - activation_norm_acc: Accumulated input norms (requires hook)
        - error_norm_acc: Accumulated output error norms (requires hook)

        Contract:
            - Safe to call even if values.grad is None (no-op)
            - Accumulates into existing EMA (doesn't reset)
        """
        if self.values.grad is None:
            return

        # Compute per-block Frobenius norms of gradients [R, K]
        grad_norms = compute_gradient_frobenius_norms(self.values.grad)

        # Update gradient EMA using TopologyScorer
        scorer = TopologyScorer(self.R, self.C, self.K, ema_alpha=self.score_ema_alpha)

        with torch.no_grad():
            self.block_score_ema = scorer.update_gradient_ema(grad_norms, self.block_score_ema)

        self._acc_steps += 1

    def score_step(self) -> None:
        """T032: Level 1 update: normalize accumulators and increment ages.

        Call every ~10 training steps. Actions:
        - Normalize activation_norm_acc by step count
        - Normalize error_norm_acc by step count
        - Increment block_age for all active blocks
        - Reset step counter

        Contract:
            - Does NOT reset block_score_ema (kept for Level 2)
            - Does NOT change topology
        """
        with torch.no_grad():
            if self._acc_steps > 0:
                # Normalize accumulated norms by step count
                self.activation_norm_acc = self.activation_norm_acc / self._acc_steps
                self.error_norm_acc = self.error_norm_acc / self._acc_steps

            # Increment block ages
            self.block_age = self.block_age + 1

        # Reset step counter (but NOT score EMA - that's kept for Level 2)
        self._acc_steps = 0

    def sync_topology_scores(self) -> None:
        """T082: Synchronize topology scores across DDP ranks.

        All-reduces block_score_ema, activation_norm_acc, and error_norm_acc
        across all ranks using average reduction. This ensures all ranks have
        identical score inputs before making topology decisions.

        Only performs sync if torch.distributed is initialized.

        Contract:
            - No-op if not in distributed training
            - Uses ReduceOp.SUM followed by division for averaging
            - Safe to call multiple times
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()

        # All-reduce the scoring buffers
        with torch.no_grad():
            # block_score_ema [R, K]
            dist.all_reduce(self.block_score_ema, op=dist.ReduceOp.SUM)
            self.block_score_ema.div_(world_size)

            # activation_norm_acc [C]
            dist.all_reduce(self.activation_norm_acc, op=dist.ReduceOp.SUM)
            self.activation_norm_acc.div_(world_size)

            # error_norm_acc [R]
            dist.all_reduce(self.error_norm_acc, op=dist.ReduceOp.SUM)
            self.error_norm_acc.div_(world_size)

    def get_topology_checksum(self) -> int:
        """T087: Get checksum of current topology for divergence detection.

        Returns a hash/checksum of the current col_indices tensor that can be
        logged and compared across ranks to detect topology divergence.

        Returns:
            Integer checksum derived from col_indices

        Contract:
            - Read-only (doesn't modify state)
            - Deterministic: same col_indices produces same checksum
            - Fast: O(R*K) computation
        """
        # Use sum of indices multiplied by position for simple checksum
        # This catches both value changes and reordering
        with torch.no_grad():
            # Flatten and compute weighted sum
            flat = self.col_indices.flatten().long()
            positions = torch.arange(len(flat), device=flat.device, dtype=torch.long)
            # Use modular arithmetic to keep checksum in reasonable range
            weighted_sum = ((flat + 1) * (positions + 1)).sum().item()
            # Also include shape info
            checksum = int(weighted_sum) ^ (self.R << 20) ^ (self.K << 10)
            return checksum

    def topology_step(
        self,
        generator: Optional[torch.Generator] = None,
        save_snapshot: bool = True,
        global_step: Optional[int] = None,
    ) -> TopologyDecisionResult:
        """T033: Level 2 update: make topology decisions (prune/grow blocks).

        Call every ~100 training steps. Actions:
        1. T083: Sync scores across DDP ranks (if distributed)
        2. Score existing blocks by gradient EMA
        3. Score candidates by activation x error product
        4. Apply epsilon-greedy exploration
        5. Select top-K per row
        6. Swap low-scoring blocks for high-scoring candidates
        7. Initialize new block weights
        8. Reset all accumulators

        Args:
            generator: RNG for exploration. If None and global_step is provided,
                a deterministic generator is created for DDP consistency.
            save_snapshot: If True, save before/after col_indices to history (T077)
            global_step: T084: If provided, creates deterministic RNG with seed
                42 + global_step for consistent topology decisions across DDP ranks.

        Returns:
            TopologyDecisionResult with swap statistics

        Contract:
            - Maintains exactly K active blocks per row
            - New blocks initialized with Kaiming x0.1 scale
            - Resets block_age to 0 for new blocks
            - All accumulators reset after decision
            - T083: Scores synced across ranks before decision in DDP
            - T084: Deterministic topology when global_step provided
        """
        import torch.distributed as dist

        # T083: Sync scores across DDP ranks before making decisions
        if dist.is_initialized():
            self.sync_topology_scores()

        # T084: Create deterministic generator if global_step provided
        if generator is None and global_step is not None:
            generator = torch.Generator(device=self.col_indices.device)
            generator.manual_seed(42 + global_step)

        # T077: Save before topology snapshot
        before_indices = self.col_indices.clone() if save_snapshot else None

        scorer = TopologyScorer(
            R=self.R,
            C=self.C,
            K=self.K,
            ema_alpha=self.score_ema_alpha,
            exploration_epsilon=0.05,  # Default from research.md
            swap_threshold=1.5,  # Default from research.md
        )

        # Compute candidate scores
        candidate_scores = scorer.compute_candidate_scores(
            self.activation_norm_acc, self.error_norm_acc
        )

        # Select new topology
        new_col_indices, pruned_positions, grown_columns = scorer.select_top_k(
            current_scores=self.block_score_ema,
            candidate_scores=candidate_scores,
            col_indices=self.col_indices,
            generator=generator,
        )

        # Initialize new block weights (T034)
        self._initialize_new_blocks(grown_columns, pruned_positions, new_col_indices)

        # Update topology
        with torch.no_grad():
            self.col_indices.copy_(new_col_indices.to(torch.int32))

        # Reset all accumulators
        with torch.no_grad():
            self.block_score_ema.zero_()
            self.activation_norm_acc.zero_()
            self.error_norm_acc.zero_()
        self._acc_steps = 0

        # T035: Compute result
        num_swaps = len(pruned_positions)
        total_blocks = self.R * self.K
        swap_rate = num_swaps / total_blocks if total_blocks > 0 else 0.0

        # T074: Track swap rate history
        self._swap_rate_history.append(swap_rate)
        if len(self._swap_rate_history) > self._swap_rate_history_max_size:
            self._swap_rate_history.pop(0)

        # T077: Save after topology snapshot
        if save_snapshot and before_indices is not None:
            self._topology_history.append((before_indices, self.col_indices.clone()))
            if len(self._topology_history) > self._topology_history_max_size:
                self._topology_history.pop(0)

        return TopologyDecisionResult(
            num_swaps=num_swaps,
            swap_rate=swap_rate,
            pruned_positions=pruned_positions,
            grown_columns=grown_columns,
        )

    def _initialize_new_blocks(
        self,
        grown_columns: List[int],
        pruned_positions: List[Tuple[int, int]],
        new_col_indices: Tensor,
    ) -> None:
        """T034: Initialize weights for newly added blocks with Kaiming x 0.1.

        Args:
            grown_columns: List of new column indices added (for reference)
            pruned_positions: List of (row, slot) that were pruned and need reinitialization
            new_col_indices: The new column indices tensor [R, K]
        """
        if not pruned_positions:
            return

        # Kaiming initialization parameters
        fan_in = self.K * self.tile_size
        gain = nn.init.calculate_gain("relu")
        std = gain / (fan_in**0.5) * 0.1  # Scale by 0.1

        with torch.no_grad():
            for row, slot in pruned_positions:
                # This slot was pruned, reinitialize its weights
                nn.init.normal_(self.values[row, slot], mean=0.0, std=std)
                # Reset age for this block
                self.block_age[row, slot] = 0

    def get_topology_stats(self) -> TopologyStats:
        """Get current topology statistics for logging.

        Returns:
            TopologyStats with density, scores, ages, entropy

        Contract:
            - Read-only (doesn't modify state)
            - Safe to call at any time
        """
        scorer = TopologyScorer(self.R, self.C, self.K)

        return TopologyStats(
            density=self.K / self.C,
            avg_block_score=self.block_score_ema.mean().item(),
            avg_block_age=self.block_age.float().mean().item(),
            column_entropy=scorer.compute_column_entropy(self.col_indices),
            num_blocks=self.R * self.K,
        )

    def get_block_age_distribution(self) -> Dict[int, int]:
        """Get histogram of block ages.

        Returns distribution of block ages across all active blocks.
        Useful for monitoring topology diversity and turnover.

        Returns:
            Dict mapping age (int) to count (int)

        Contract:
            - Read-only (doesn't modify state)
            - Safe to call at any time
            - Ages are in topology steps (Level 2 updates)
        """
        # Get block_age tensor [R, K] and compute histogram
        ages = self.block_age.flatten().tolist()

        # Build histogram as dict
        histogram: Dict[int, int] = {}
        for age in ages:
            age_int = int(age)
            histogram[age_int] = histogram.get(age_int, 0) + 1

        return histogram

    def get_avg_swap_rate(self) -> float:
        """Get rolling average of swap rates over recent topology steps.

        Returns:
            Average swap rate (0.0 to 1.0), or 0.0 if no history

        Contract:
            - Read-only (doesn't modify state)
            - Uses rolling window of last 100 topology steps
        """
        if not self._swap_rate_history:
            return 0.0
        return sum(self._swap_rate_history) / len(self._swap_rate_history)

    def get_topology_history(self) -> List[Tuple[Tensor, Tensor]]:
        """Get saved topology snapshots.

        Returns:
            List of (before_col_indices, after_col_indices) tuples
            from recent topology_step() calls

        Contract:
            - Returns copies of the history (read-only)
            - Limited to last 10 snapshots
        """
        return list(self._topology_history)

    def get_density(self) -> float:
        """Get actual current density.

        Returns:
            K / C (should match configured density)
        """
        return self.K / self.C

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return state for checkpointing.

        Includes:
            - values: Weight parameters
            - col_indices: Topology
            - bias: Bias parameters (if present)
            - block_score_ema: Scoring state
            - block_age: Block ages
            - accumulators: Accumulated norms
            - _acc_steps: Accumulator step counter
            - _swap_rate_history: Rolling swap rate history
        """
        # Use parent's state_dict which handles parameters and buffers
        state = super().state_dict(*args, **kwargs)
        # Add extra non-tensor state
        state["_acc_steps"] = self._acc_steps
        state["_swap_rate_history"] = list(self._swap_rate_history)
        # Note: topology_history is intentionally not saved (can be large)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load state from checkpoint.

        Restores full layer state including topology and scoring.

        Args:
            state_dict: State dictionary from state_dict()
            strict: If True, raise error on missing/unexpected keys
        """
        # Extract extra state before parent load (which may modify dict)
        acc_steps = state_dict.pop("_acc_steps", 0)
        swap_rate_history = state_dict.pop("_swap_rate_history", [])

        # Load parameters and buffers
        super().load_state_dict(state_dict, strict=strict)

        # Restore extra state
        self._acc_steps = acc_steps
        self._swap_rate_history = list(swap_rate_history)
        # Reset topology history on load (not saved)
        self._topology_history = []

    def to_dense(self) -> Tensor:
        """Convert current topology to dense weight matrix.

        Returns:
            Dense weight tensor [out_features, in_features]

        Use for:
            - Debugging / visualization
            - Comparison with nn.Linear
            - Export to frameworks without sparse support
        """
        from .block_ell import to_dense as block_ell_to_dense

        return block_ell_to_dense(
            values=self.values,
            col_indices=self.col_indices,
            R=self.R,
            C=self.C,
            K=self.K,
            B=self.tile_size,
        )

    @classmethod
    def from_dense(
        cls,
        dense_layer: nn.Linear,
        tile_size: int = 16,
        density: float = 0.5,
        score_ema_alpha: float = 0.95,
    ) -> "CMSBlockLinear":
        """Create sparse layer from existing dense layer.

        Initializes topology by magnitude-based pruning of dense weights.
        Uses Frobenius norm of each block to select the top-K most important
        blocks per row, preserving the highest magnitude weights.

        Args:
            dense_layer: Source nn.Linear layer
            tile_size: Block size for sparse format
            density: Target density (fraction of blocks to keep per row)
            score_ema_alpha: EMA momentum for gradient scores (default 0.95)

        Returns:
            CMSBlockLinear initialized from dense weights

        Raises:
            ValueError: If dense layer dimensions not divisible by tile_size

        Example:
            >>> dense = nn.Linear(128, 256)
            >>> sparse = CMSBlockLinear.from_dense(dense, tile_size=16, density=0.5)
            >>> # sparse now has topology based on dense weight magnitudes
        """
        from .block_ell import from_dense as block_ell_from_dense

        in_features = dense_layer.in_features
        out_features = dense_layer.out_features
        has_bias = dense_layer.bias is not None

        # Validate dimensions
        if in_features % tile_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tile_size ({tile_size})"
            )
        if out_features % tile_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tile_size ({tile_size})"
            )

        # Convert dense weights to block-ELL format using magnitude-based selection
        dense_weight = dense_layer.weight.data  # [out_features, in_features]
        values, col_indices, R, C, K, B = block_ell_from_dense(
            dense_weight, tile_size=tile_size, density=density
        )

        # Create the sparse layer (this will initialize with random topology)
        sparse_layer = cls(
            in_features=in_features,
            out_features=out_features,
            tile_size=tile_size,
            density=density,
            bias=has_bias,
            score_ema_alpha=score_ema_alpha,
            device=dense_layer.weight.device,
            dtype=dense_layer.weight.dtype,
        )

        # Override with the values and topology from magnitude-based selection
        with torch.no_grad():
            sparse_layer.values.copy_(values)
            sparse_layer.col_indices.copy_(col_indices.to(torch.int32))

            # Copy bias if present
            if has_bias:
                sparse_layer.bias.copy_(dense_layer.bias.data)

        return sparse_layer

    def extra_repr(self) -> str:
        """String representation for print(layer)."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tile_size={self.tile_size}, density={self.density:.2f}, "
            f"K={self.K}, bias={self.bias is not None}"
        )
