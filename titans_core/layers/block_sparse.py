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

from .block_ell import BlockELLConfig, BlockELLTensor


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
        self._block_ell_config = BlockELLConfig(
            R=self.R, C=self.C, K=self.K, B=tile_size
        )

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
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
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

        # Initialize weights and topology
        self._reset_parameters()
        self._initialize_topology()

    def _reset_parameters(self) -> None:
        """Initialize weight parameters with Kaiming uniform."""
        # Scale factor for sparse initialization
        # Use fan_in based on actual connections, not full dense
        fan_in = self.K * self.tile_size
        gain = nn.init.calculate_gain("relu")
        std = gain / (fan_in ** 0.5)
        nn.init.normal_(self.values, mean=0.0, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _initialize_topology(self) -> None:
        """Initialize column indices with random topology."""
        # Random unique columns per row
        for r in range(self.R):
            perm = torch.randperm(self.C, device=self.col_indices.device)
            self.col_indices[r] = perm[: self.K].to(torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: block-sparse matrix multiplication.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor with same batch/seq dims, out_features last

        Contract:
            - Output shape matches nn.Linear contract
            - Numerically equivalent to dense @ sparse_mask at initialization
            - Supports 2D and 3D input tensors
        """
        # Handle 2D vs 3D input
        input_shape = x.shape
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
        output = torch.zeros(
            batch_size, seq_len, self.R, B,
            dtype=x.dtype, device=x.device
        )

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
            block_outputs = torch.einsum('bski,koi->bsko', input_gathered, weights)
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
        """Accumulate gradient statistics for importance scoring.

        Call after backward() each step. Updates:
        - block_score_ema: EMA of gradient Frobenius norms
        - activation_norm_acc: Accumulated input norms (requires hook)
        - error_norm_acc: Accumulated output error norms (requires hook)

        Contract:
            - Safe to call even if values.grad is None (no-op)
            - Accumulates into existing EMA (doesn't reset)
        """
        raise NotImplementedError("CMSBlockLinear.accumulate_scores not yet implemented")

    def score_step(self) -> None:
        """Level 1 update: normalize accumulators and increment ages.

        Call every ~10 training steps. Actions:
        - Normalize activation_norm_acc by step count
        - Normalize error_norm_acc by step count
        - Increment block_age for all active blocks
        - Reset step counter

        Contract:
            - Does NOT reset block_score_ema (kept for Level 2)
            - Does NOT change topology
        """
        raise NotImplementedError("CMSBlockLinear.score_step not yet implemented")

    def topology_step(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> TopologyDecisionResult:
        """Level 2 update: make topology decisions (prune/grow blocks).

        Call every ~100 training steps. Actions:
        1. Score existing blocks by gradient EMA
        2. Score candidates by activation x error product
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
            - New blocks initialized with Kaiming x0.1 scale
            - Resets block_age to 0 for new blocks
            - All accumulators reset after decision
        """
        raise NotImplementedError("CMSBlockLinear.topology_step not yet implemented")

    def get_topology_stats(self) -> TopologyStats:
        """Get current topology statistics for logging.

        Returns:
            TopologyStats with density, scores, ages, entropy

        Contract:
            - Read-only (doesn't modify state)
            - Safe to call at any time
        """
        raise NotImplementedError("CMSBlockLinear.get_topology_stats not yet implemented")

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
        """
        # Use parent's state_dict which handles parameters and buffers
        state = super().state_dict(*args, **kwargs)
        # Add any extra state needed
        state["_acc_steps"] = self._acc_steps
        return state

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True
    ) -> None:
        """Load state from checkpoint.

        Restores full layer state including topology and scoring.
        """
        # Extract extra state
        acc_steps = state_dict.pop("_acc_steps", 0)

        # Load parameters and buffers
        super().load_state_dict(state_dict, strict=strict)

        # Restore extra state
        self._acc_steps = acc_steps

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
    ) -> "CMSBlockLinear":
        """Create sparse layer from existing dense layer.

        Initializes topology by magnitude-based pruning of dense weights.

        Args:
            dense_layer: Source nn.Linear layer
            tile_size: Block size for sparse format
            density: Target density

        Returns:
            CMSBlockLinear initialized from dense weights
        """
        raise NotImplementedError("CMSBlockLinear.from_dense not yet implemented")

    def extra_repr(self) -> str:
        """String representation for print(layer)."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tile_size={self.tile_size}, density={self.density:.2f}, "
            f"K={self.K}, bias={self.bias is not None}"
        )
